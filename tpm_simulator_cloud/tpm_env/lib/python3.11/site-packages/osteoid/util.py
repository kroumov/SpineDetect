from .skeleton import Skeleton

def load(filename:str) -> Skeleton:
	if filename.endswith("swc"):
		with open(filename, "rt") as f:
			data = f.read()
		return Skeleton.from_swc(data)

	with open(filename, "rb") as f:
		data = f.read()
	return Skeleton.from_precomputed(data)

def save(filename:str, skel:Skeleton):
	if filename.endswith("swc"):
		binary = skel.to_swc()
	else:
		binary = skel.to_precomputed()

	with open(filename, "wb") as f:
		f.write(binary)
