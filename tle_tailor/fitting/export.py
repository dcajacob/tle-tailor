"""Write fitted SGP4 elements into fixed-width TLE records."""
from sgp4.exporter import export_tle as _export_tle
from sgp4.io import compute_checksum


def export_tle(model):
    """Encode very small drag values using the smallest TLE exponent."""
    line1, line2 = _export_tle(model)
    if len(line1) != 69 and 0 < abs(model.bstar) < 1e-10:
        # sgp4 emits a literal 'e-10' (or smaller) in the B* column.
        # Keep all other fields from that record. With exponent -9, the
        # implied-decimal mantissa resolves adjacent values 1e-14 apart.
        import re
        field = re.match(r"[ -]\d{5}e-\d{2,}(?= )", line1[53:])
        if field:
            units = round(abs(model.bstar) * 1e14)
            sign = '-' if model.bstar < 0 and units else ' '
            data = line1[:53] + f'{sign}{units:05d}-9' + line1[53 + field.end():-1]
            line1 = data + str(compute_checksum(data))
    if len(line1) != 69 or len(line2) != 69:
        raise ValueError("Fitted elements cannot be encoded in two 69-character TLE records")
    return line1, line2
