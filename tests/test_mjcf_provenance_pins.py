"""The MJCF provenance pins must describe the XML that is actually on disk.

There are two independent pins in this repo for the same fact -- which MJCF
revision the numbers were produced on:

  * ``retarget.mjcf_provenance.MJCF_COMMIT``  (smoke 5 / smoke 6 restamp path)
  * ``grounding.collect_clean_demos.MJCF_COMMIT`` + ``MJCF_SHA256`` (smoke 4)

Wave 3.5-E landed the finger-proximal ``<pair>`` block in 1b0f13f and moved the
working-tree XML to that commit's blob.  The smoke-5 pin stayed on 4508421, so
**every** smoke-5 rerun restamped its artefacts with 4508421 and attached a false
"working-tree XML does not match" guard -- and nothing complained, because that
path is internally self-consistent: it says 4508421, it hashes 4508421's blob,
and it stamps 4508421.

That is the point of this file.  Asserting the two pins are equal would have
stayed green through every round of that drift -- two pins that go stale
*together* satisfy an equality check perfectly.  The check with catching power is
declared vs computed: the pin has to equal the sha of the XML on disk.  That
goes false the moment either pin drifts, and it also catches the case the
cross-pin equality misses entirely, where both pins are moved but the generator
is never re-run.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from grounding.collect_clean_demos import MJCF_COMMIT as DEMOS_COMMIT  # noqa: E402
from grounding.collect_clean_demos import MJCF_SHA256 as DEMOS_SHA256  # noqa: E402
from retarget import mjcf_provenance as mp  # noqa: E402
from retarget.mjcf_provenance import blob_sha256  # noqa: E402
from retarget.retarget_smoke import SCENE_XML, sha256_file  # noqa: E402


def test_both_pins_name_the_xml_on_disk():
    """declared == computed, for the smoke-4 pin and for the smoke-5 pin."""
    on_disk = sha256_file(SCENE_XML)
    rel = SCENE_XML.relative_to(REPO).as_posix()

    assert DEMOS_SHA256 == on_disk, (
        "grounding.collect_clean_demos.MJCF_SHA256 is stale: it declares "
        f"{DEMOS_SHA256[:16]} but grounding/assets/allegro_floating.xml is "
        f"{on_disk[:16]}. Re-run the generator or move the pin -- do not "
        "restamp."
    )
    assert blob_sha256(DEMOS_COMMIT, rel) == on_disk, (
        f"collect_clean_demos.MJCF_COMMIT ({DEMOS_COMMIT}) does not own the "
        f"XML on disk ({on_disk[:16]})"
    )
    assert mp.MJCF_COMMIT == DEMOS_COMMIT, (
        f"the two pins disagree: mjcf_provenance says {mp.MJCF_COMMIT}, "
        f"collect_clean_demos says {DEMOS_COMMIT}. They describe the same XML "
        "and must not be able to drift apart silently."
    )
    assert blob_sha256(mp.MJCF_COMMIT, rel) == on_disk, (
        f"mjcf_provenance.MJCF_COMMIT ({mp.MJCF_COMMIT}) does not own the XML "
        f"on disk ({on_disk[:16]})"
    )
    assert blob_sha256(mp.MJCF_COMMIT_BASELINE, rel) != on_disk, (
        "the baseline pin now names the current XML, so 'differs from baseline' "
        "carries no information any more"
    )


def test_working_tree_scene_is_not_edited_on_top_of_a_pinned_commit():
    """The stamp's own claim, recomputed rather than trusted."""
    prov = mp.mjcf_provenance()
    assert prov["scene_matches_mjcf_commit"], prov.get("provenance_guard")
    assert prov["scene_differs_from_baseline"]


def test_committed_smoke5_artefact_matches_its_own_stamp():
    """The artefact on disk names a commit its own bytes were produced on."""
    import json

    report = json.loads(mp.REPORT_PATH.read_text())
    stamp = report["mjcf_commit"]
    assert stamp["mjcf_commit"] == mp.MJCF_COMMIT
    assert stamp["scene_sha256"] == sha256_file(SCENE_XML)
    assert "provenance_guard" not in stamp, stamp.get("provenance_guard")
    qa = json.loads(mp.QA_PATH.read_text())
    assert qa["physics_provenance"]["mjcf_commit"] == mp.MJCF_COMMIT
