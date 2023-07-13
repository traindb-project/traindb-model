"""
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import argparse
import io
import json
import logging
from multiprocessing import Process
import os
from pathlib import Path
import signal
import sys
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import jaydebeapi
import jpype
import pandas as pd
import psutil
import uvicorn

from TrainDBBaseModelRunner import TrainDBModelRunner

app = FastAPI()
training_processes = []

root_dir = Path(__file__).resolve().parent.parent
lib_dir = root_dir.joinpath('lib').joinpath('*')
modeltype_dir = root_dir.joinpath('models')
model_dir = root_dir.joinpath('trained_models')

def train_model(modeltype_class, model_name, jdbc_driver_class,
                db_url, db_user, db_pwd, select_training_data_sql, metadata):
    jpype.startJVM(classpath = str(lib_dir), convertStrings=True)
    conn = jaydebeapi.connect(jdbc_driver_class, db_url, [ db_user, db_pwd ])
    curs = conn.cursor()
    curs.execute(select_training_data_sql)
    header = [desc[0] for desc in curs.description]
    training_data = pd.DataFrame(curs.fetchall(), columns=header)

    modeltype_path = get_modeltype_path(modeltype_class)
    model_path = get_model_path(model_name)
    runner = TrainDBModelRunner()
    model = runner._train(modeltype_class, modeltype_path, training_data, metadata)

    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.save(model_path)

def get_modeltype_path(modeltype_name: str):
    return str(modeltype_dir.joinpath(modeltype_name+'.py'))

def get_model_path(model_name: str):
    return str(model_dir.joinpath(model_name))

@app.get("/")
def hello():
    return "Hello, TrainDB!"

@app.get("/modeltype/{modeltype_class}/hyperparams")
async def list_hyperparameters(modeltype_class: str):
    runner = TrainDBModelRunner()
    modeltype_path = get_modeltype_path(modeltype_class)
    return json.dumps(runner._hyperparams(modeltype_class, modeltype_path))

@app.post("/modeltype/{modeltype_class}/train")
async def train(
        modeltype_class: str,
        model_name: str,
        jdbc_driver_class: str,
        db_url: str,
        db_user: str,
        db_pwd: str,
        select_training_data_sql: str,
        metadata_file: UploadFile = File(...)):

    metadata = json.loads(metadata_file.file.read())
    p = Process(target=train_model,
                args=(modeltype_class, model_name, jdbc_driver_class,
                      db_url, db_user, db_pwd, select_training_data_sql, metadata))
    p.start()
    training_processes.append({"pid": p.pid, "model": model_name})

    return {"message": "Start training model '" + model_name + "' @ PID " + str(p.pid)}

def convert_str(s: str):
    if len(s.strip()) == 0:
        return ""
    return s.strip()

@app.get("/model/{model_name}/synopsis")
async def synopsis(
        modeltype_class: str,
        model_name: str,
        rows: int):
    modeltype_path = get_modeltype_path(modeltype_class)
    model_path = get_model_path(model_name)

    runner = TrainDBModelRunner()
    syn_data = runner._synthesize(
            modeltype_class, modeltype_path, model_path, rows)

    response = StreamingResponse(io.StringIO(syn_data.to_csv(index=False)), media_type="text/csv")
    return response

@app.get("/model/{model_name}/infer")
async def infer(
        modeltype_class: str,
        model_name: str,
        agg_expr: str,
        group_by_column: Optional[str] = "",
        where_condition: Optional[str] = ""):
    modeltype_path = get_modeltype_path(modeltype_class)
    model_path = get_model_path(model_name)
    grp_by = convert_str(group_by_column)
    cond = convert_str(where_condition)

    runner = TrainDBModelRunner()
    aqp_result, confidence_interval = runner._infer(
            modeltype_class, modeltype_path, model_path, agg_expr, grp_by, cond)

    cols = []
    if (len(grp_by) > 0):
        cols.append(grp_by)
    cols.append(agg_expr)
    df = pd.DataFrame(aqp_result, columns=cols)
    response = StreamingResponse(io.StringIO(df.to_csv(index=False)), media_type="text/csv")
    return response

@app.get("/status/")
def status():
    res = []
    for proc in training_processes:
        try:
            p = psutil.Process(proc["pid"])
            status = p.status()
        except psutil.NoSuchProcess:
            status = "finished"
        res.append({"model": proc["model"], "status": status})
    return res

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0',
                        help='IP address of the TrainDB Model Server REST API')
    parser.add_argument('--port', default='8000',
                        help='port of the TrainDB Model Server REST API')
    parser.add_argument('--ssl_keyfile', nargs='?', default='')
    parser.add_argument('--ssl_certfile', nargs='?', default='')

    parser.add_argument('--log_level', type=int, default=logging.DEBUG)
    parser.add_argument('--log_dir', default=str(root_dir.joinpath('logs')))
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logfile_format = args.log_dir + "/traindb_modelserver_{}.log".format(time.strftime("%Y%m%d"))
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(logfile_format),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    # ignore the SIGCHLD signal to avoid zombie processes
    # it doesn't care to know when child processes exit
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    # prerequisite: pip install fastapi uvicorn python-multipart
    # testing: launch browser with "http://0.0.0.0:58080" then see hello message
    if len(args.ssl_keyfile) > 0 and len(args.ssl_certfile) > 0:
        uvicorn.run(app, host=args.host, port=int(args.port),
                    ssl_keyfile=args.ssl_keyfile, ssl_certfile=args.ssl_certfile)
    else:
        uvicorn.run(app, host=args.host, port=int(args.port))

    sys.exit("Shutting down, bye bye!")

