# Manual Git History Cleanup

The current working change stops tracking checkpoint binaries and the detailed
workstation inventory while preserving local working-tree copies. Earlier commits
still contain those objects. Rewriting shared history is intentionally not part of
the automated hardening pass.

Before proceeding, obtain maintainer approval, make an access-controlled artifact
backup of every required model, record verified SHA-256 values, notify all
collaborators, pause merges, and create a remote backup. Then use a fresh mirror
clone rather than the development checkout:

```bash
git clone --mirror https://github.com/johnelyahchouchi/UAV-AI-ROS2-PROTOTYPE.git UAV-AI-ROS2-PROTOTYPE-security-cleanup.git
cd UAV-AI-ROS2-PROTOTYPE-security-cleanup.git
git filter-repo --path-glob '*.pt' --path-glob '*.pth' --path-glob '*.engine' --path '00_PROJECT_GUIDE/ORIGINAL_SOURCE_INVENTORY.csv' --invert-paths
git count-objects -vH
git log --all -- '00_PROJECT_GUIDE/ORIGINAL_SOURCE_INVENTORY.csv'
git rev-list --objects --all | grep -E '\.(pt|pth|engine)$'
```

Review the rewritten mirror and obtain explicit approval before any force update.
Only then, during a coordinated maintenance window, update the remote:

```bash
git push --force --mirror origin
```

Every collaborator must discard old clones/forks or carefully rebase onto the new
history. Rotate any secret discovered during the review; history removal is not a
substitute for rotation. Decide separately whether historical ONNX artifacts or
path-bearing run metadata represent sensitive IP before adding them to the filter.
