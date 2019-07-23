#!/usr/bin/env bash
set -e
## FIXME : xselはサーバー上ではうごかないぜ。

python build.py
cat .build/script.py | xsel --clipboard --input
echo 'copied to clipboard'
