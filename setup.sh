#!/bin/bash
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
deep_imitation_path=$(cut -d ' ' -f2 <<< $(cut -d':' -f2 <<< $(pip show deep_imitation|grep Location)))
ln -sf $here/experimental $deep_imitation_path/deep_imitation --verbose

