import itertools

from schrodinger.adapter import to_structure
from schrodinger.application.matsci.nano.xtal import get_cov_radii

RADII = get_cov_radii()
at_nums = [*range(1, 21), *range(31, 39), *range(49, 57), *range(81, 84)]


st = to_structure("N#N")
st.generate3dConformation()
for at1, at2 in itertools.combinations(at_nums, 2):
    st.atom[1].atomic_number = at1
    st.atom[2].atomic_number = at2
    parity = (at1 + at2) % 2
    start_dist = max(0.5, (RADII[at1] + RADII[at2]) * 0.6)
    dist = start_dist
    while dist <= 8:
        st.adjust(dist, 1, 2)
        # run the bottom two spin states for each, cations and anions are opposite parity
        for spin_mod in (1, 3):
            st.write(
                f"{st.atom[1].element}_{st.atom[2].element}_{round(dist, 2)}_0_{parity+spin_mod}.mae"
            )
            st.write(
                f"{st.atom[1].element}_{st.atom[2].element}_{round(dist, 2)}_1_{1-parity+spin_mod}.mae"
            )
            st.write(
                f"{st.atom[1].element}_{st.atom[2].element}_{round(dist, 2)}_-1_{1-parity+spin_mod}.mae"
            )
        if dist < 6.0:
            dist += 0.1
        else:
            dist += 0.2
