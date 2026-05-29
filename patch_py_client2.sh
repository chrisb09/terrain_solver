#!/bin/bash
sed -i '/if app_id != 0:/,/pass/d' /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/dl_clients/phydll_dl_client.py
