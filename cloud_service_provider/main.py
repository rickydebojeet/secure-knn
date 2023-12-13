from flask import Flask
from flask import request
import json
import sqlite3
import pickle
import numpy as np
from keirudb import Keiru

db_file = "db.pickle"
qry_db = "qry.db"

qrs_connection = None
qrs_cursor = None
db = None
keirudb = Keiru()

app = Flask(__name__)


def init():
    global db
    global qrs_connection
    global qrs_cursor
    qrs_connection = sqlite3.connect(qry_db, check_same_thread=False)
    qrs_cursor = qrs_connection.cursor()

    qrs_cursor.execute(
        """CREATE TABLE IF NOT EXISTS queries
                (userid text PRIMARY KEY, Mt text)"""
    )

    try:
        with open(db_file, "rb") as f:
            db = pickle.load(f)
    except:
        pass


def create_temp_db(query_id, M_t):
    global db
    if len(db) == 0:
        return False

    M_t_inv = np.linalg.inv(M_t)

    n = db.shape[1]

    temp_db = []
    for p_dash in db:
        p_dash = p_dash.reshape(1, n)
        p_dash_dash = np.dot(p_dash, M_t_inv).reshape(n)
        temp_db.append(p_dash_dash)

    temp_db = np.array(temp_db)
    print("temp_db: \n", temp_db)
    keirudb.add(query_id, temp_db)
    return True


def check_temp_db(query_id):
    global keirudb
    if keirudb.getdata(query_id) is not None:
        return True
    qrs_cursor.execute("SELECT * FROM queries WHERE userid=?", (query_id,))
    row = qrs_cursor.fetchone()
    if row is None:
        return False
    M_t = np.array(json.loads(row[1]))
    keirudb.startpush(query_id)
    create_temp_db(query_id, M_t)
    return True


@app.route("/clear_db", methods=["POST"])
def clear_db():
    global db
    db = None
    with open(db_file, "wb") as f:
        pickle.dump(db, f)
    print("OK: Database cleared successfully")
    return "OK"


@app.route("/upload", methods=["POST"])
def upload():
    global db
    data = request.get_json()
    datapoints = np.array(data["datapoints"])
    if db is None:
        db = datapoints
    elif len(db[0]) != len(datapoints[0]):
        return "ERROR: Dimension mismatch", 400
    else:
        db = np.vstack((db, datapoints))
    with open(db_file, "wb") as f:
        pickle.dump(db, f)

    print("OK: Data received successfully")
    # print(db)
    return "OK: Data received successfully"


@app.route("/get_data", methods=["GET"])
def get_data():
    global db
    print("OK: Data sent successfully")
    return json.dumps(db.tolist())


@app.route("/send_query", methods=["POST"])
def send_query():
    data = request.get_json()
    query_id = data["query_id"]
    M_t = np.array(data["M_t"])
    print("M_t: \n", M_t)
    qrs_cursor.execute(
        "INSERT OR REPLACE INTO queries VALUES (?, ?)", (query_id, str(M_t.tolist()))
    )
    qrs_connection.commit()
    keirudb.startpush(query_id)
    create_temp_db(query_id, M_t)
    return "OK"


@app.route("/compute_kNN", methods=["POST"])
def compute_kNN():
    """Compute kNN for a given query"""
    data = request.get_json()
    query_id = data["query_id"]
    query = np.array(data["query"])
    k = data["k"]
    # print("query: ", query)
    print("query_id: ", query_id)
    if check_temp_db(query_id) is False:
        return "ERROR: Query ID not found", 400
    temp_db = keirudb.getdata(query_id)
    if temp_db is None:
        return "ERROR: Query ID not found", 400
    if len(temp_db) == 0:
        return "ERROR: Database is empty", 400
    if len(temp_db[0]) != len(query):
        return "ERROR: Dimension mismatch", 400
    dist = np.dot(temp_db, query)
    print("dist: ", dist)
    index = np.argpartition(dist, k)[:k]
    # print(index)
    # print("db[index]: ", db[index])
    return json.dumps({"datapoints": db[index].tolist()})


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=8888, debug=True)
