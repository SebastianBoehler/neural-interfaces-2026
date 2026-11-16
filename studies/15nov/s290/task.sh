#!/bin/bash
# s290 part 2 task wrapper: md5 of the code files at the top of the job log, then the driver (arg: task id or smoke)
echo "code md5 ($(pwd)):"; md5sum experiments/deep.py experiments/dg.py experiments/run_cv.py experiments/session_bn.py experiments/protocol.py studies/12nov/ttt5.py studies/13nov/ttt7.py studies/13nov/ttt18.py studies/15nov/tsm288/spd.py studies/15nov/s290/ttt290.py
echo "cell SCHEME=$SCHEME PSEED=$PSEED DRAW=$DRAW FAMS=$FAMS BASE_ONLY=$BASE_ONLY PROTOS=$PROTOS"
exec python -I studies/15nov/s290/ttt290.py "$@"
