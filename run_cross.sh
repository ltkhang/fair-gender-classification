which python

python cross_validate.py ./tmp_checkpoints/resnet34/fairface/model 18 fairface  all/resnet34_fairface_fairface
python cross_validate.py tmp_checkpoints/balance/fairface/model 18 fairface all/balance_fairface_fairface
python cross_validate.py tmp_checkpoints/ms1m/fairface/model 27 fairface all/ms1m_fairface_fairface

python cross_validate.py tmp_checkpoints/resnet34/fairface/model 18 utkface all/resnet34_fairface_utkface
python cross_validate.py tmp_checkpoints/balance/fairface/model 18 utkface all/balance_fairface_utkface
python cross_validate.py tmp_checkpoints/ms1m/fairface/model 27  utkface all/ms1m_fairface_utkface

python cross_validate.py tmp_checkpoints/resnet34/fairface/model 18 adience all/resnet34_fairface_adience
python cross_validate.py tmp_checkpoints/balance/fairface/model 18 adience all/balance_fairface_adience
python cross_validate.py tmp_checkpoints/ms1m/fairface/model 27 adience all/ms1m_fairface_adience

python cross_validate.py tmp_checkpoints/resnet34/utkface/model 19 fairface  all/resnet34_utkface_fairface
python cross_validate.py tmp_checkpoints/balance/utkface/model 20 fairface all/balance_utkface_fairface
python cross_validate.py tmp_checkpoints/ms1m/utkface/model 19 fairface all/ms1m_utkface_fairface

python cross_validate.py tmp_checkpoints/resnet34/utkface/model 19 utkface  all/resnet34_utkface_utkface
python cross_validate.py tmp_checkpoints/balance/utkface/model 20 utkface all/balance_utkface_utkface
python cross_validate.py tmp_checkpoints/ms1m/utkface/model 19 utkface all/ms1m_utkface_utkface

python cross_validate.py tmp_checkpoints/resnet34/utkface/model 19 adience  all/resnet34_utkface_adience
python cross_validate.py tmp_checkpoints/balance/utkface/model 20 adience all/balance_utkface_adience
python cross_validate.py tmp_checkpoints/ms1m/utkface/model 19 adience all/ms1m_utkface_adience