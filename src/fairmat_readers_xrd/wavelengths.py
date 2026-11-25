"""Characteristic Kα1/Kα2 reference wavelengths in angstroms.

International Tables for Crystallography (2006), Vol. C, Table 4.2.2.1:
https://onlinelibrary.wiley.com/iucr/itc/Cb/ch4o2v0001/sec4o2o2/
Values retain the published precision; uncertainty estimates are not stored.

XRAY_WAVELENGTHS provides fallback values for the RAW4 parser.
"""

K_ALPHA_WAVELENGTHS = {
    'Cr': (2.2897260, 2.2936510),
    'Fe': (1.9360410, 1.9399730),
    'Co': (1.7889960, 1.7928350),
    'Cu': (1.54059290, 1.54442740),
    'Ga': (1.3401270, 1.3440260),
    'Mo': (0.70931715, 0.713607),
    'Ag': (0.55942178, 0.5638131),
    'In': (0.5121251, 0.5165572),
}

# Kα references are shared with NM. RAW4 fallbacks also include Kβ wavelengths
# and a Kα2/Kα1 intensity ratio of 0.5.
XRAY_WAVELENGTHS = {
    material: {
        'K_alpha1': K_ALPHA_WAVELENGTHS[material][0],
        'K_alpha2': K_ALPHA_WAVELENGTHS[material][1],
        'K_beta': k_beta,
        'K_alpha_avg': (
            2 * K_ALPHA_WAVELENGTHS[material][0] + K_ALPHA_WAVELENGTHS[material][1]
        )
        / 3,
        'K_alpha2_K_alpha1_ratio': 0.5,
    }
    for material, k_beta in {
        'Cu': 1.392250,
        'Mo': 0.632305,
        'Co': 1.620830,
        'Fe': 1.757462,
        'Cr': 2.084920,
        'Ag': 0.497082,
    }.items()
}
