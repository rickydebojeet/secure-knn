import pandas as pd
import numpy as np
import pickle
import requests
from flask import Flask, request
import json

C = 5
E = 5
B_2 = 2.0

key_file = "key_file.pickle"
cloud_provider = "http://localhost:8888"

# np.random.seed(54)

db = {}

app = Flask(__name__)


def init():
    """Initialize the database."""
    global db
    try:
        with open(key_file, "rb") as f:
            db = pickle.load(f)
    except:
        pass


def load_data(file_path, headings, delim=","):
    """Load data from file."""
    df = pd.read_table(
        file_path,
        sep=delim,
        names=headings,
    )
    print("OK: Data loaded successfully")
    return df


def Class_value(Class):
    if Class == 2:
        return 1
    else:
        return 0


def process_data(df):
    """Process data."""
    df["Bare_Nuclei"][df["Bare_Nuclei"] == "?"] = "0"
    df["Bare_Nuclei"] = df["Bare_Nuclei"].astype(str).astype(int)
    df = df.drop(["Sample_code_number"], axis=1)
    df["Class"] = df["Class"].apply(Class_value)

    X = np.array(df.iloc[:, :9])
    y = np.array(df["Class"])

    print("OK: Data processed successfully")

    return X, y


def generate_invertible_matrix(dim):
    while True:
        M = np.random.randint(-2, 2, (dim, dim))
        if np.linalg.det(M) != 0:
            return M
        
def generate_M_t(n, q_max, Max_norm):
    while True:
        M_t = np.random.randint(q_max + 1, q_max + 1 + 5, size=(n, n))
        diagonal = np.random.randint(int(Max_norm + 1), int(Max_norm + 1) + 10, size=n)
        np.fill_diagonal(M_t, diagonal)
        if np.linalg.det(M_t) != 0:
            return M_t


def save_data(d, m, c, e, n, M_base, M_base_inv, s_vector, w_vector, max_norm):
    """Save data to file."""
    global db
    db["d"] = d
    db["m"] = m
    db["c"] = c
    db["e"] = e
    db["n"] = n
    db["M_base"] = M_base
    db["Mbase_inv"] = M_base_inv
    db["s_vector"] = s_vector
    db["w_vector"] = w_vector
    db["Max_norm"] = max_norm
    with open(key_file, "wb") as f:
        pickle.dump(db, f)

    print("OK: Data saved successfully")


def encrypt_data(database):
    """Encrypt data."""
    d = database.shape[1]
    m = database.shape[0]
    c = C
    e = E
    n = d + 1 + c + e

    s_vector = np.random.randint(-15, 15, (1, d + 1))
    w_vector = np.random.randint(-10, 10, (1, c))
    M_base = generate_invertible_matrix(n)
    M_base_inv = np.linalg.inv(M_base)

    database_dash = []
    Max_norm = 0

    for p in database:
        # find max eucleadian norm
        if Max_norm < np.linalg.norm(p):
            Max_norm = np.linalg.norm(p)

        secret_vector = s_vector[0, :d] - 2 * p
        secret_vector = np.array(secret_vector).reshape(1, d)

        one_vector = s_vector[0, d] + np.sum(p**2)
        one_vector = np.array(one_vector).reshape(1, 1)

        z_vector = np.random.randint(-2, 2, (1, e))

        p_dash = np.concatenate((secret_vector, one_vector, w_vector, z_vector), axis=1)

        p_dash = np.dot(p_dash, M_base_inv).reshape(n)
        database_dash.append(p_dash)

    save_data(d, m, c, e, n, M_base, M_base_inv, s_vector, w_vector, Max_norm)

    print("OK: Data encrypted successfully")

    return np.array(database_dash)


def decrypt_data(database_dash):
    global db
    M_base = db["M_base"]
    s_vector = db["s_vector"]
    d = db["d"]
    n = db["n"]
    decrypted_database = []
    for p_dash in database_dash:
        p_dash = p_dash.reshape(1, n)
        p_tilda = np.dot(p_dash, M_base)

        p_tilda = s_vector[0, :d] - p_tilda[0, :d]
        p_tilda = (p_tilda / 2).round().reshape(d)

        decrypted_database.append(p_tilda)

    decrypted_database = np.array(decrypted_database)
    print(decrypted_database)
    return decrypted_database
    # we need to do something here


def clear_datacenter_data():
    response = requests.post(cloud_provider + "/clear_db")
    if response.status_code != 200:
        print("Error clearing datacenter database")
        return False
    return True


def upload_datacenter_data(encrypted_data: np.array):
    response = requests.post(
        cloud_provider + "/upload", json={"datapoints": encrypted_data.tolist()}
    )
    if response.status_code != 200:
        print("Error uploading datacenter database")
        return False
    return True


def download_datacenter_data():
    response = requests.get(cloud_provider + "/get_data")
    if response.status_code != 200:
        print("Error downloading datacenter database")
        return False
    return np.array(response.json())


def process_query(q_dot):
    global db
    M_base = db["M_base"]
    Max_norm = db["Max_norm"]
    n = db["n"]
    d = db["d"]
    c = db["c"]
    e = db["e"]

    q_max = np.max(q_dot)

    M_t = generate_M_t(n, q_max, Max_norm)
    M_sec = np.dot(M_t, M_base)

    query_vector = q_dot.reshape(1, d)
    one_vector = np.array([1]).reshape(1, 1)
    x_vector = np.random.randint(-3, 3, (1, c))
    zero_vector = np.zeros((1, e))

    q_dash = np.concatenate((query_vector, one_vector, x_vector, zero_vector), axis=1)
    q_dash = q_dash.reshape(n)
    q_nn = np.diag(q_dash)

    # sample error matrix of n * n dimensions which contains random real numbers larger than q_max
    E = np.random.randint(int(q_max + 1), int(q_max + 1) + 10, size=M_t.shape)

    # encrypted query
    q_cap = B_2 * (np.dot(q_nn, M_sec) + E)

    return (q_cap, M_t)


@app.route("/encrypt_query", methods=["POST"])
def encrypt_query():
    data = request.get_json()
    q_dot = np.array(data["datapoints"])
    query_id = data["query_id"]
    # print("q_dot: ", q_dot)
    print("query_id: ", query_id)
    q_cap, M_t = process_query(q_dot)
    # print("q_cap: ", q_cap)

    # print("M_t: \n", M_t)

    response = requests.post(
        cloud_provider + "/send_query", json={"query_id": query_id, "M_t": M_t.tolist()}
    )

    if response.status_code != 200:
        print("Error pushing query to datacenter")
        return "Error pushing query to datacenter", 400

    return json.dumps({"datapoints": q_cap.tolist()})


@app.route("/decrypt_data", methods=["POST"])
def decrypt():
    data = request.get_json()
    datapoints = np.array(data["datapoints"])
    decrypted_points = decrypt_data(datapoints)
    return json.dumps({"datapoints": decrypted_points.tolist()})


@app.route("/upload", methods=["POST"])
def upload():
    """Upload data to cloud provider."""
    file = "breast-cancer-wisconsin.data"
    headings = [
        "Sample_code_number",
        "Clump_Thickness",
        "Uniformity_of_Cell_Size",
        "Uniformity_of_Cell_Shape",
        "Marginal_Adhesion",
        "Single_Epithelial_Cell_Size",
        "Bare_Nuclei",
        "Bland_Chromatin",
        "Normal_Nucleoli",
        "Mitoses",
        "Class",
    ]
    df = load_data(file, headings)

    X, y = process_data(df)

    print("Original data: \n", X)

    database_dash = encrypt_data(X)
    print("Encrypted data: \n", database_dash)
    if not clear_datacenter_data():
        return "Error clearing datacenter database", 400
    if not upload_datacenter_data(database_dash):
        return "Error uploading datacenter database", 400
    print("OK: Data uploaded to cloud provider successfully")
    return "OK: Data uploaded to cloud provider successfully"


@app.route("/download", methods=["GET"])
def download():
    database_dash = download_datacenter_data()
    if database_dash is False:
        return "Error downloading datacenter database", 400
    decrypted_database = decrypt_data(database_dash)
    # need to think something here
    return "It works"


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", port=8080, debug=True)
