#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2021 The TARTRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

def load_server_json(load_file):
    with open(load_file, 'r') as f:
        server_dict = json.load(f)
    return server_dict


def parse_server(json_file):
    server_dict = load_server_json(load_file=json_file)
    world_size = server_dict['world_size']

    with open('/proc/sys/kernel/hostname', 'r') as f:
        hostname = f.readline().strip()

    find_server = False
    for server_now in server_dict['servers']:
        if server_now['name'] != hostname:
            continue
        find_server = True
        gpus = server_now['gpus']
        local_size = server_now['local_size']
        start_rank = server_now['start_rank']
    assert find_server, '{} not exist!'.format(hostname)
    assert len(gpus) >= local_size, 'gpu number should larger than local size'
    assert local_size >= 1, 'local size should larger than 1'
    gpus = gpus[:local_size]

    return hostname, gpus, world_size, local_size, start_rank

