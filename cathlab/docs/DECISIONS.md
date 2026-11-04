# vcathlab decisions

### V1: A separate app, linked to vstent only by a dataset format (2026-11-04)
The human asked for the virtual cath lab as a different app in a new folder, "so we don't mix
it, and it can help us produce results for our models". `cathlab/` is its own package
(`vcathlab`) with its own environment, tests and checks; it never imports vstent and vstent
never imports it. Data moves through `docs/DATA_FORMAT.md`. An independent simulator also
means a vstent model is never checked by its own code. Alternatives: a vstent sub-package
(mixes the two), a separate git repository (more overhead now; still possible later).

### V2: Copied from vstent: the C-arm convention and the X-ray chain's ranges (2026-11-04)
`geometry.py` is vstent's `imaging/projection.py` (convention D49) trimmed, so DICOM written
here is read by vstent as intended. The imaging-chain ranges (contrast attenuation per mm,
background, blur, edge enhancement, noise) are vstent's ARCADE-fitted values (D39). Both
evolve here on their own from now on. The D49 angle convention is still unconfirmed on real
two-view DICOM; sharing it means the cath lab cannot detect an error in it.

### V3: Attenuation adds where vessels cross; touching lumens join (2026-11-04)
Real X-ray attenuation adds along a ray, so a crossing looks darker: a cue the detector must
learn to tell from a lesion. vstent's renderer took the longest chord. Here contributions add,
except where the two 3D centreline points are closer than the sum of their radii plus 0.2 mm
(the lumens touch: a branch point, two pieces of one vessel), where the larger counts. Exact for
separate round lumens; approximate where lumens merge. Vessels are drawn in 15 mm pieces so a
vessel crossing itself also adds.

### V4: Anatomical trees in the heart's grooves, with published normal sizes (2026-11-04)
Random curves on a sphere (vstent P5, P8) do not overlap the way real arteries do. Here the
arteries follow the AV and interventricular grooves of a spheroid heart oriented as in the
chest, so standard projections give realistic overlaps and foreshortening, and every lesion has
a SYNTAX segment. Proximal diameters: left main 4.5 mm, proximal LAD 3.7 mm, proximal RCA 3.9
mm in right and 2.8 mm in left dominance (Dodge et al., Circulation 1992), times a patient
scale (SD 10%); circumflex and side branches by Murray's law. Groove positions, branch angles
and lengths are sampling choices and may be tuned.
