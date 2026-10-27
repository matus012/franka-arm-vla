#!/bin/bash
# git archive HEAD -> PERUN $PROJ/repos/<sha> + symlink $PROJ/repo, sha256-verified. Never rsyncs the working tree. Run from anywhere inside the repo.
set -euo pipefail
PROJ=/mnt/project/<project-id>/113_vla_tabletop
H=<cluster-host>
cd "$(git rev-parse --show-toplevel)"
SHA=$(git rev-parse --short HEAD)
T=$(mktemp -d)/repo.tgz
git archive --format=tar.gz -o "$T" HEAD
L=$(sha256sum "$T" | cut -d' ' -f1)
scp -o BatchMode=yes "$T" $H:$PROJ/repo.tgz
R=$(ssh -o BatchMode=yes $H "sha256sum $PROJ/repo.tgz" | cut -d' ' -f1)
[ "$L" = "$R" ] || { echo "sha256 MISMATCH $L $R"; exit 1; }
# Atomic swap: each sha unpacks once to repos/<sha> (never rewritten: same sha = same archive); `repo` is a symlink flipped with mv -T, so a running job (whose bind
# resolved the old target at start) keeps its tree. A plain-directory `repo` (pre-symlink layout) is removed once.
ssh -o BatchMode=yes $H "cd $PROJ && mkdir -p repos && { [ -d repos/$SHA ] || { rm -rf repos/$SHA.tmp && mkdir repos/$SHA.tmp   && tar xzf repo.tgz -C repos/$SHA.tmp && echo $SHA > repos/$SHA.tmp/GIT_HEAD && mv repos/$SHA.tmp repos/$SHA; }; }   && { [ -L repo ] || rm -rf repo; } && ln -sfn repos/$SHA repo.new && mv -Tf repo.new repo"
echo "pushed HEAD $SHA sha256 $L"
