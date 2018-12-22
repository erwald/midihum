from collections import OrderedDict

import numpy as np

# Dictionary of chords qualities and chords (in integer notation).
# Taken from https://github.com/yuma-m/pychord/blob/master/pychord/constants/qualities.py .
QUALITY_MAP = OrderedDict((
    # chords consist of 2 notes
    ('5', [0, 7]),
    ('m3', [0, 3]),
    ('M3', [0, 4]),
    # 3 notes
    ('maj', [0, 4, 7]),
    ('min', [0, 3, 7]),
    ('dim', [0, 3, 6]),
    ('aug', [0, 4, 8]),
    ('sus2', [0, 2, 7]),
    ('sus4', [0, 5, 7]),
    # 4 notes
    ('6', [0, 4, 7, 9]),
    ('7', [0, 4, 7, 10]),
    ('7-5', [0, 4, 6, 10]),
    ('7+5', [0, 4, 8, 10]),
    ('7sus4', [0, 5, 7, 10]),
    ('m7', [0, 3, 7, 10]),
    ('m7-5', [0, 3, 6, 10]),
    ('dim6', [0, 3, 6, 9]),
    ('M7', [0, 4, 7, 11]),
    ('M7+5', [0, 4, 8, 11]),
    ('mM7', [0, 3, 7, 11]),
    ('add9', [0, 4, 7, 14]),
    ('add11', [0, 4, 7, 17]),
    # 5 notes
    ('6/9', [0, 4, 7, 9, 14]),
    ('9', [0, 4, 7, 10, 14]),
    ('m9', [0, 3, 7, 10, 14]),
    ('M9', [0, 4, 7, 11, 14]),
    ('9sus4', [0, 5, 7, 10, 14]),
    ('7-9', [0, 4, 7, 10, 13]),
    ('7+9', [0, 4, 7, 10, 15]),
    ('11', [0, 7, 10, 14, 17]),
    ('7+11', [0, 4, 7, 10, 18]),
    ('7-13', [0, 4, 7, 10, 20]),
    # 6 notes
    ('13', [0, 4, 7, 10, 14, 21]),
))

# Dictionary of chords qualities and their attributes.
QUALITY_ATTRIBUTES_MAP = OrderedDict((
    # chords consist of 2 notes
    ('5', ('indeterminate', 'dyad')),
    ('m3', ('minor', 'dyad')),
    ('M3', ('major', 'dyad')),
    # 3 notes
    ('maj', ('major', 'triad')),
    ('min', ('minor', 'triad')),
    ('dim', ('diminished', 'triad')),
    ('aug', ('augmented', 'triad')),
    ('sus2', ('suspended', 'triad')),
    ('sus4', ('suspended', 'triad')),
    # 4 notes
    ('6', ('major', 'seventh')),
    ('7', ('diminished', 'seventh')),
    ('7-5', ('diminished', 'seventh')),
    ('7+5', ('augmented', 'seventh')),
    ('7sus4', ('suspended', 'seventh')),
    ('m7', ('minor', 'seventh')),
    ('m7-5', ('diminished', 'seventh')),
    ('dim6', ('diminished', 'seventh')),
    ('M7', ('major', 'seventh')),
    ('M7+5', ('augmented', 'seventh')),
    ('mM7', ('minor', 'seventh')),
    ('add9', ('major', 'ninth')),
    ('add11', ('major', 'eleventh')),
    # 5 notes
    ('6/9', ('major', 'ninth')),
    ('9', ('major', 'ninth')),
    ('m9', ('minor', 'ninth')),
    ('M9', ('major', 'ninth')),
    ('9sus4', ('suspended', 'ninth')),
    ('7-9', ('major', 'ninth')),
    ('7+9', ('major', 'ninth')),
    ('11', ('major', 'eleventh')),
    ('7+11', ('major', 'eleventh')),
    ('7-13', ('major', 'thirteenth')),
    # 6 notes
    ('13', ('major', 'thirteenth')),
))


def clamp_into_octave(chord):
    # Put root at 0.
    chord_with_root_at_zero = chord - np.amin(chord)

    # Put all notes within a single octave, removing duplicates and sorting in
    # ascending order.
    clamped_chord = np.unique(np.mod(chord_with_root_at_zero, 12))

    return clamped_chord


def get_inversions(chord):
    inversions = []

    for i in range(len(chord)):
        inversion = np.concatenate((chord[i:, ], chord[:i, ] + 12))
        # Move root down to 0 again.
        inversion = inversion - np.amin(inversion)
        inversions.append(inversion)

    return inversions


def chord_quality_for_inversion(chord):
    """ Finds the quality of the given chord.

    Based on https://github.com/yuma-m/pychord/blob/master/pychord/analyzer.py .

    :param list[int] chord: a chord in integer notation, e.g. [0, 4, 7]
    :rtype: str|None
    """
    # Go through the list of chords in reverse order (because we want to
    # encounter the more specific chord first, in case there are multiple
    # matches), and return the quality of the first match.
    for quality, notes in reversed(QUALITY_MAP.items()):
        if np.array_equal(chord, clamp_into_octave(notes)):
            return quality

    return None


def attributes_for_chord_quality(chord_quality):
    for quality, attributes in QUALITY_ATTRIBUTES_MAP.items():
        if quality == chord_quality:
            return attributes

    return None


def chord_attributes(chord):
    ''' Finds the quality of the given chord, including any inversions of it.
    Returns a tuple of the first match, with chord attributes.

    Based on https://github.com/yuma-m/pychord/blob/master/pychord/analyzer.py .

    :param list[int] chord: a chord in integer notation, e.g. [0, 4, 7]
    :rtype: (str,str)|None
    '''
    if len(chord) < 2:
        return None

    # Make sure root is at 0, and that all notes fit within the octave. Iow,
    # that all note values are between 0 and 11, sorted in ascending order.
    chord = clamp_into_octave(chord)

    for notes in get_inversions(chord):
        quality = chord_quality_for_inversion(notes)
        if quality is not None:
            return attributes_for_chord_quality(quality)

    return None
