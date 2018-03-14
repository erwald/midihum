def exists(it):
    return (it is not None)


def map_removing_none(function, sequence):
    return filter(exists, map(function, sequence))
