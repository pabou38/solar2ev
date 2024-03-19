
:: comment with rem will be printed. not with ::

echo 'run multiples training with synthetic data. results log to file'
echo 'run in conda venv'

:: -a for classification

::goto :continue

rem "=======> retrain on larger dataset (ie from concatenate) <========="
:: -w to swap order
python -m synthetic_train -r
python -m synthetic_train -r -w

::rem "retrain and shuffle"
python -m synthetic_train -r -s
python -m synthetic_train -r -s -w

:continue

goto :end

rem "=======> continue training <========"

:: all model trainable
python -m synthetic_train -c
python -m synthetic_train -c -w

:: freeze LSTM, replace last classifier
python -m synthetic_train -c -n
python -m synthetic_train -c -n -w

:: freeze LSTM, unfreeze last classifier
python -m synthetic_train -c -u
python -m synthetic_train -c -u -w

:end

echo 'sdv done'