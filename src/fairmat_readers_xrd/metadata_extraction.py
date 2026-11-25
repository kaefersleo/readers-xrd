#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Read sample, instrument, and scan metadata and record where each value was read."""

import hashlib
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import Element

import numpy as np
import pint

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

    from fairmat_readers_xrd.ikz import BRMLfile, RASXfile

ureg = pint.get_application_registry()


def _text(value: object) -> str | None:
    """Strip whitespace from text and return None for blank or non-string values."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def _key(prefix: str, value: object) -> str:
    """Build a record key from a readable label and a hash of the original value."""
    label = re.sub(r'[^a-zA-Z0-9_]+', '_', str(value)).strip('_')[:40]
    digest = hashlib.sha256(str(value).encode()).hexdigest()[:10]
    return f'{prefix}_{label}_{digest}'


def _record(
    key: str, values: dict[str, Any], sources: Mapping[str, str | list[str]]
) -> dict[str, Any]:
    """
    Build a metadata record with its source paths.

    Args:
        key (str): Identifier used to group matching records.
        values (dict): Metadata fields and their values.
        sources (dict): Source path or list of paths for each retained field.

    Returns:
        dict: The record key, values other than None or blank text, and lists of paths.
    """
    values = {
        name: value
        for name, value in values.items()
        if value is not None and not (isinstance(value, str) and not value.strip())
    }
    return {
        'key': key,
        'values': values,
        'sources': {
            name: sources[name] if isinstance(sources[name], list) else [sources[name]]
            for name in values
        },
    }


def _same(left: Any, right: Any) -> bool:
    """Compare metadata values exactly after converting compatible quantities."""
    if left is None or right is None:
        return left is right
    if hasattr(left, 'units'):
        if not hasattr(right, 'units') or not left.is_compatible_with(right):
            return False
        return np.array_equal(left.magnitude, right.to(left.units).magnitude)
    return np.array_equal(left, right)


def _merge_records_by_key(
    records: list[dict[str, Any]], logger: 'BoundLogger | None' = None
) -> list[dict[str, Any]]:
    """Merge records by key and keep fields whose values agree across all records."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record['key'], []).append(record)
    output = []
    for key, entries in groups.items():
        values, sources = {}, {}
        names = set().union(*(entry['values'] for entry in entries))
        for name in sorted(names):
            candidates = [entry['values'].get(name) for entry in entries]
            if not all(_same(candidates[0], value) for value in candidates[1:]):
                if logger is not None:
                    logger.warning(
                        f'Omitting {key}/{name} because values differ or are missing.'
                    )
                continue
            values[name] = candidates[0]
            sources[name] = list(
                dict.fromkeys(
                    path for entry in entries for path in entry['sources'].get(name, [])
                )
            )
        if values:
            output.append({'key': key, 'values': values, 'sources': sources})
    return output


def _parse_metadata_quantity(
    text: str | float | None,
    unit: str | None,
    path: str,
    logger: 'BoundLogger | None',
    compatible_unit: str | None = None,
) -> pint.Quantity | None:
    """
    Read an optional metadata value as a finite Pint quantity.

    Args:
        text: Numeric value or its representation in the source file.
        unit (str): Unit reported in the source file.
        path (str): Source path to include in warnings.
        logger (BoundLogger or None): Logger for invalid values, if provided.
        compatible_unit (str, optional): Unit the value must be compatible with.

    Returns:
        pint.Quantity or None: The reported quantity, or None if it is missing
            or invalid.
    """
    if text is None or not unit:
        return None
    try:
        value = float(text) * ureg(unit)
        if not np.isfinite(value.magnitude):
            raise ValueError('Non-finite value')
        if compatible_unit and not value.is_compatible_with(compatible_unit):
            raise ValueError('Incompatible unit')
        return value
    except (ValueError, TypeError, pint.PintError):
        if logger is not None:
            logger.warning(f'Ignoring invalid optional metadata at {path}.')
        return None


def xrdml_metadata(
    root: Element,
    measurement: Element,
    scans: list[Element],
    ns: dict[str, str],
    logger: 'BoundLogger | None' = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract reported sample, scan, and instrument metadata from XRDML.

    Args:
        root: Root element of the XRDML document.
        measurement: Measurement element that contains the scans and beam paths.
        scans: Scan elements in the order they appear in the measurement.
        ns (dict): Namespace mapping used to find XRDML elements.
        logger (BoundLogger, optional): Logger for invalid or conflicting metadata.

    Returns:
        dict: Metadata records grouped by category, with source paths for
            each retained field.
    """
    details: dict[str, list[dict[str, Any]]] = {
        'users': [],
        'notes': [],
        'instrument_notes': [],
        'program': [],
        'detector': [],
    }
    sample_name = _text(root.findtext('xrdml:sample/xrdml:name', namespaces=ns))
    if sample_name:
        details['sample'] = [
            _record(
                'sample',
                {'sample_name': sample_name},
                {'sample_name': 'sample/name'},
            )
        ]
    for index, scan in enumerate(scans):
        base = f'xrdMeasurement/scan[{index + 1}]/header'
        author = _text(
            scan.findtext('xrdml:header/xrdml:author/xrdml:name', namespaces=ns)
        )
        if author:
            details['users'].append(
                _record(
                    _key('user', author),
                    {'name': author},
                    {'name': f'{base}/author/name'},
                )
            )
        program = scan.find('xrdml:header/xrdml:source/xrdml:applicationSoftware', ns)
        details['program'].append(
            _record(
                'acquisition_program',
                {
                    'name': _text(program.text) if program is not None else None,
                    'version': _text(program.get('version'))
                    if program is not None
                    else None,
                },
                {
                    'name': f'{base}/source/applicationSoftware',
                    'version': f'{base}/source/applicationSoftware/@version',
                },
            )
        )
        control = scan.find(
            'xrdml:header/xrdml:source/xrdml:instrumentControlSoftware', ns
        )
        if control is not None and (control_name := _text(control.text)):
            control_text = f'Instrument control software: {control_name}'
            if version := _text(control.get('version')):
                control_text += f' (version {version})'
            details['instrument_notes'].append(
                _record(
                    _key('control_software', control_text),
                    {'text': control_text},
                    {'text': f'{base}/source/instrumentControlSoftware'},
                )
            )
    # Omit the shared acquisition program if any scan lacks a program name.
    programs = details['program']
    if any(not record['values'].get('name') for record in programs):
        if any(record['values'] for record in programs) and logger is not None:
            logger.warning(
                'Incomplete scan acquisition program; omitting common metadata.'
            )
        details['program'] = []
    for parent, base in ((root, ''), (measurement, 'xrdMeasurement/')):
        for index, node in enumerate(parent.findall('xrdml:comment/xrdml:entry', ns)):
            text = _text(node.text)
            if text:
                details['notes'].append(
                    _record(
                        _key('comment', f'{base}{index}'),
                        {'text': text},
                        {'text': f'{base}comment/entry[{index + 1}]'},
                    )
                )
    aperture_tags = {
        'mask': 'width',
        'antiScatterSlit': 'height',
        'divergenceSlit': 'height',
        'receivingSlit': 'height',
    }
    for side in ('incidentBeamPath', 'diffractedBeamPath'):
        beam = measurement.find(f'xrdml:{side}', ns)
        if beam is None:
            continue
        occurrences: dict[str, int] = {}
        for node in beam:
            tag = node.tag.rsplit('}', 1)[-1]
            occurrences[tag] = occurrences.get(tag, 0) + 1
            index = occurrences[tag]
            base = f'xrdMeasurement/{side}/{tag}[{index}]'
            identity = node.get('id') or str(index)
            key = _key(f'{side}_{tag}', identity)
            name = _text(node.get('name'))
            if tag == 'detector':
                details['detector'].append(
                    _record(
                        'primary_detector',
                        {'description': name},
                        {'description': f'{base}/@name'},
                    )
                )
            elif tag in (
                'monochromator',
                'sollerSlit',
                *aperture_tags,
                'xRayMirror',
                'parallelPlateCollimator',
                'beamAttenuator',
            ):
                values: dict[str, Any] = {'name': name}
                sources: dict[str, str | list[str]] = {'name': f'{base}/@name'}
                if tag == 'monochromator':
                    values['chemical_formula'] = _text(
                        node.findtext('xrdml:crystal', namespaces=ns)
                    )
                    sources['chemical_formula'] = f'{base}/crystal'
                    hkl = [
                        node.findtext(f'xrdml:hkl/xrdml:{axis}', namespaces=ns)
                        for axis in ('h', 'k', 'l')
                    ]
                    if all(value is not None for value in hkl):
                        try:
                            values['reflection'] = np.array(
                                [int(value) for value in hkl], dtype=np.int64
                            )
                            sources['reflection'] = [
                                f'{base}/hkl/{axis}' for axis in ('h', 'k', 'l')
                            ]
                        except ValueError:
                            if logger is not None:
                                logger.warning(
                                    f'Ignoring invalid reflection at {base}/hkl.'
                                )
                    family = 'monochromators'
                elif tag == 'sollerSlit' or tag in aperture_tags:
                    if tag == 'sollerSlit':
                        child_name = 'opening'
                        field = 'soller_angle'
                        unit = 'radian'
                        family = 'collimators'
                    else:
                        child_name = aperture_tags[tag]
                        field = 'size'
                        unit = 'meter'
                        family = 'apertures'
                    child = node.find(f'xrdml:{child_name}', ns)
                    values[field] = _parse_metadata_quantity(
                        child.text if child is not None else None,
                        child.get('unit') if child is not None else None,
                        f'{base}/{child_name}',
                        logger,
                        unit,
                    )
                    sources[field] = f'{base}/{child_name}'
                    values['description'] = (
                        f'{tag} in {side}. Opening field: {child_name}.'
                    )
                    sources['description'] = base
                elif tag == 'beamAttenuator':
                    values['material'] = _text(
                        node.findtext('xrdml:material', namespaces=ns)
                    )
                    sources['material'] = f'{base}/material'
                    family = 'attenuators'
                else:
                    family = 'mirrors' if tag == 'xRayMirror' else 'collimators'
                details.setdefault(family, []).append(_record(key, values, sources))
    return {
        family: _merge_records_by_key(records, logger)
        for family, records in details.items()
    }


def rasx_metadata(
    reader: 'RASXfile', logger: 'BoundLogger | None' = None
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract metadata from the measurement conditions of RASX profiles.

    Fixed axis positions are included when all profiles report them with
    compatible units. The positions follow the original profile order.

    Args:
        reader (RASXfile): Parsed RASX file with profiles and their metadata paths.
        logger (BoundLogger, optional): Logger for invalid or conflicting metadata.

    Returns:
        dict: Metadata records grouped by category, with source paths for
            each retained field.
    """
    details: dict[str, list[dict[str, Any]]] = {
        'users': [],
        'notes': [],
        'instrument_notes': [],
        'detector': [],
        'sample': [],
        'positioners': [],
    }
    profiles = reader.meta[: len(reader.data)]
    for index, meta in enumerate(profiles):
        base = reader.metadata_paths[index]
        general = meta.get('GeneralInformation', {})
        operator = _text(general.get('Operator'))
        if operator:
            details['users'].append(
                _record(
                    _key('user', operator),
                    {'name': operator},
                    {'name': f'{base}/GeneralInformation/Operator'},
                )
            )
        for name in ('Comment', 'Memo', 'UserGroup', 'SystemName', 'Type', 'Version'):
            value = general.get(name)
            if value is not None and str(value).strip():
                text = str(value) if name in ('Comment', 'Memo') else f'{name}: {value}'
                details['notes'].append(
                    _record(
                        _key('rasx_note', f'{name}:{text}'),
                        {'text': text},
                        {'text': f'{base}/GeneralInformation/{name}'},
                    )
                )
        details['sample'].append(
            _record(
                'sample',
                {'sample_name': _text(general.get('SampleName'))},
                {'sample_name': f'{base}/GeneralInformation/SampleName'},
            )
        )
        hardware = meta.get('HardwareConfig', {})
        details['detector'].append(
            _record(
                'primary_detector',
                {'description': _text(hardware.get('Detector'))},
                {
                    'description': (
                        f'{base}/HWConfigurations/Categories/Detector/@SelectedUnit'
                    )
                },
            )
        )
        for name, value in hardware.get('optics_settings', {}).items():
            if _text(value):
                text = f'Optics {name}: {value.strip()}'
                details['instrument_notes'].append(
                    _record(
                        _key('rasx_optics', f'{name}:{value}'),
                        {'text': text},
                        {'text': f'{base}/HWConfigurations/Optics/{name}'},
                    )
                )
    # Keep fixed positions for the axes below when every profile reports
    # a position in compatible angular or length units.
    axes = {
        'ThetaS',
        'ThetaD',
        'TwoTheta',
        'Omega',
        'Chi',
        'Phi',
        'Z',
        'Zs',
        'Zr',
        'Ts',
        'Alpha',
        'Beta',
        'Rx',
        'Ry',
    }
    for name in sorted(axes):
        entries = [meta.get('Axes', {}).get(name) for meta in profiles]
        if not entries or any(
            axis is None or axis.State != 'Fixed' for axis in entries
        ):
            continue
        values = [
            _parse_metadata_quantity(
                axis.Position,
                axis.Unit,
                f'{reader.metadata_paths[index]}/Axes/{name}/@Position',
                logger,
            )
            for index, axis in enumerate(entries)
        ]
        if any(
            value is None
            or not (
                value.is_compatible_with('radian') or value.is_compatible_with('meter')
            )
            for value in values
        ):
            continue
        unit = values[0].units
        if not all(value.is_compatible_with(unit) for value in values):
            continue
        details['positioners'].append(
            _record(
                _key('rasx_axis', name),
                {
                    'name': name,
                    'description': 'Fixed axis positions in RASX profile order.',
                    'value': np.array([value.to(unit).magnitude for value in values])
                    * unit,
                },
                {
                    'name': (
                        f'{reader.metadata_paths[0]}/Axes/Axis[@Name="{name}"]/@Name'
                    ),
                    'description': (
                        f'{reader.metadata_paths[0]}/Axes/Axis[@Name="{name}"]/@State'
                    ),
                    'value': [
                        f'{reader.metadata_paths[i]}/Axes/'
                        f'Axis[@Name="{name}"]/@Position'
                        for i in range(len(entries))
                    ],
                },
            )
        )
    return {
        family: _merge_records_by_key(records, logger)
        for family, records in details.items()
    }


def _as_list(
    value: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Wrap a single XML entry in a list and use an empty list for missing entries."""
    return value if isinstance(value, list) else [value] if value else []


def brml_metadata(
    reader: 'BRMLfile', logger: 'BoundLogger | None' = None
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract fixed drive positions and instrument metadata from BRML.

    Drive positions are included when every scan provides values with
    compatible units. The positions follow the order of the RawData files.

    Args:
        reader (BRMLfile): Parsed BRML file with the drive and instrument records.
        logger (BoundLogger, optional): Logger for invalid or conflicting metadata.

    Returns:
        dict: Metadata records grouped by category, with source paths for
            each retained field.
    """
    details: dict[str, list[dict[str, Any]]] = {'positioners': [], 'detector': []}
    for name, entries in reader.motors.items():
        if len(entries) != len(reader.data['ScanName']) or any(
            entry is None for entry in entries
        ):
            continue
        positions = [
            _parse_metadata_quantity(
                entry.get('Value'), entry.get('Unit'), f'Drives/{name}/Position', logger
            )
            for entry in entries
        ]
        if any(
            value is None
            or not (
                value.is_compatible_with('radian') or value.is_compatible_with('meter')
            )
            for value in positions
        ):
            continue
        unit = positions[0].units
        if not all(value.is_compatible_with(unit) for value in positions):
            continue
        details['positioners'].append(
            _record(
                _key('brml_motor', name),
                {
                    'name': name,
                    'description': (
                        'Fixed drive positions recorded in RawData block order.'
                    ),
                    'value': np.array([value.to(unit).magnitude for value in positions])
                    * unit,
                },
                {
                    'name': (
                        f'FixedInformation/Drives/'
                        f'InfoData[@LogicName="{name}"]/@LogicName'
                    ),
                    'description': 'FixedInformation/Drives',
                    'value': [
                        f'{item["path"]}/FixedInformation/Drives/'
                        f'InfoData[@LogicName="{name}"]/Position/@Value'
                        for item in reader.fixed_information
                    ],
                },
            )
        )
    for item in reader.fixed_information:
        detector_count = len(details['detector'])
        for side in ('PrimaryTracks', 'SecondaryTracks'):
            tracks = item['value'].get('Instrument', {}).get(side, {})
            for track in _as_list(tracks.get('TrackInfoData')):
                for index, node in enumerate(
                    _as_list(track.get('MountedOptics', {}).get('InfoData'))
                ):
                    base = (
                        f'{item["path"]}/FixedInformation/Instrument/{side}/'
                        f'TrackInfoData[@LogicName="{track.get("@LogicName", "")}"]/'
                        f'MountedOptics/InfoData[{index + 1}]'
                    )
                    key = _key(
                        f'brml_{side}',
                        node.get('BeringInfo', {}).get('@ObjectPath')
                        or (
                            f'{track.get("@LogicName", "")}/'
                            f'{node.get("@LogicName", index)}'
                        ),
                    )
                    name = _text(node.get('@VisibleName'))
                    kind = node.get('@xsi:type', '')
                    class_path = node.get('BeringInfo', {}).get('@ClassPath', '')
                    values: dict[str, Any] = {'name': name}
                    sources = {'name': f'{base}/@VisibleName'}
                    tube = node.get('Tube', {})
                    serial = _text(tube.get('SerialNumber', {}).get('@Value'))
                    if serial:
                        details.setdefault('source_fabrication', []).append(
                            _record(
                                'tube_fabrication',
                                {'serial_number': serial},
                                {'serial_number': f'{base}/Tube/SerialNumber/@Value'},
                            )
                        )
                    if '/Optic/Detector/' in class_path:
                        details['detector'].append(
                            _record(
                                'primary_detector',
                                {'description': name},
                                {'description': f'{base}/@VisibleName'},
                            )
                        )
                        continue
                    if kind in (
                        'FixedSlitInfoDataExt',
                        'VarSlitInfoData',
                        'DetectorSlitInfoData',
                    ):
                        field = 'SlitWidth' if 'SlitWidth' in node else 'Height'
                        size = node.get(field, {})
                        values['size'] = _parse_metadata_quantity(
                            size.get('@Value'),
                            size.get('@Unit'),
                            f'{base}/{field}',
                            logger,
                            'meter',
                        )
                        sources['size'] = f'{base}/{field}/@Value'
                        values['description'] = (
                            f'Slit in {side}. Opening field: {field}.'
                        )
                        sources['description'] = base
                        family = 'apertures'
                    elif kind == 'GoebelMirrorInfoData':
                        family = 'mirrors'
                    else:
                        # Product names do not establish crystal material or geometry.
                        text = (
                            f'Mounted optic {side}/{node.get("@LogicName", index)}: '
                            f'{name or kind}; source type {kind}.'
                        )
                        details.setdefault('instrument_notes', []).append(
                            _record(key, {'text': text}, {'text': base})
                        )
                        continue
                    details.setdefault(family, []).append(_record(key, values, sources))
        if len(details['detector']) == detector_count:
            details['detector'].append(
                _record(
                    'primary_detector',
                    {'description': None},
                    {},
                )
            )
    return {
        family: _merge_records_by_key(records, logger)
        for family, records in details.items()
    }


def raw_metadata(metadata: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """
    Collect user, comment, and acquisition program metadata from parsed RAW4 data.

    Args:
        metadata (dict): Values extracted by the RAW4 parser.

    Returns:
        dict: Lists of user, note, and program records with their source tags.
    """
    details: dict[str, list[dict[str, Any]]] = {'users': [], 'notes': [], 'program': []}
    user = _text(metadata.get('user'))
    if user:
        details['users'].append(
            _record(_key('user', user), {'name': user}, {'name': 'RAW4/USER'})
        )
    comment = _text(metadata.get('comment'))
    if comment:
        details['notes'].append(
            _record('raw4_comment', {'text': comment}, {'text': 'RAW4/COMMENT'})
        )
    creator = _text(metadata.get('creator'))
    if creator:
        details['program'].append(
            _record(
                'acquisition_program',
                {'name': creator, 'version': _text(metadata.get('creator_version'))},
                {'name': 'RAW4/CREATOR', 'version': 'RAW4/CREATOR_VERSION'},
            )
        )
    return details
