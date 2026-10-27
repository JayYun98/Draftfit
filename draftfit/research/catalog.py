"""Source provenance for the selected methods; no external runtime launchers."""

SOURCES = {
    "alr_ira": ("js-lee-AI/ALR-IRA", "fa473642d48ea2667675d706ca8f16a74111183e"),
    "cast": ("js-lee-AI/CAST", "2eed73f38f9210f1d5ddf01de8059cf3ffb43c19"),
    "osprey": ("LeanModels/Osprey", "f62cd48844b03f3cb929543da25c15756e1f1095"),
    "redline": ("js-lee-AI/Redline", "d78ede9b035420f3c8c8c63c185ed55be5ea038f"),
}


def source_catalog():
    return [{"project": name, "url": "https://github.com/" + repo,
             "revision": revision, "integration": "native"}
            for name, (repo, revision) in SOURCES.items()]
