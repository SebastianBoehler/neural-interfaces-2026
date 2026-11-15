#!/bin/bash
# s285 task wrapper: md5 of the code files at the top of the job log, then ttt18 (arg: task id or smoke)
echo "code md5 ($(pwd)):"; md5sum experiments/deep.py experiments/dg.py experiments/run_cv.py experiments/session_bn.py experiments/protocol.py studies/12nov/ttt5.py studies/13nov/ttt7.py studies/13nov/ttt18.py
echo "cell SCHEME=$SCHEME PSEED=$PSEED DRAW=$DRAW FAMS=$FAMS BASE_ONLY=$BASE_ONLY PROTOS=$PROTOS"
exec python -I studies/13nov/ttt18.py "$@"
