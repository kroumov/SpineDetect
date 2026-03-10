class SkeletonUnassignedEdgeError(Exception):
  """This skeleton has an edge to a vertex that doesn't exist."""
  pass

class SkeletonDecodeError(Exception):
  """Unable to decode a binary skeleton into a Python object."""
  pass

class SkeletonEncodeError(Exception):
  """Unable to encode a PrecomputedSkeleton into a binary object."""
  pass

class SkeletonTransformError(Exception):
  """Unable to apply a spatial transfrom to the current coordinate system."""
  pass

class SkeletonAttributeMixingError(Exception):
  """
  These skeletons have different vertex attributes 
  and cannot be recombined without manual intervention.
  """
  pass
