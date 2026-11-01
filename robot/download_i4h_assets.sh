#!/usr/bin/env bash
# 手術器具の 3D モデル（NVIDIA Isaac for Healthcare, v0.9.0 staging）を robot/pickup_assets/sim/assets/i4h/ に取得する。
set -e; B=https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/v0.9.0/7cf86a1/Props
D="$(cd "$(dirname "$0")" && pwd)/pickup_assets/sim/assets/i4h"; OR='shared_OR_without_Mark/Collected_surgery_room_movie_with_heart_adjusted/Assets/Surgery Room'
for f in "SurgicalInstruments_A/Model/sm_scissors_a01_01.usd" "SurgicalInstruments_A/Model/sm_forceps_a01_01.usd" "SurgicalInstruments_A/Model/sm_scalpelshandle_a01_01.usd" "InsulinSyringe_A/Model/sm_insulinsyringe_a01_01.usd"; do
  mkdir -p "$D/$OR/$(dirname "$f")"; curl -fsSL "$B/${OR// /%20}/${f}" -o "$D/$OR/$f"; echo "OK $f"; done
