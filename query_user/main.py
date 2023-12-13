import numpy as np
import pandas as pd
import requests

data_owner = "http://localhost:8080"
cloud_provider = "http://localhost:8888"
B_1 = 50.0
query_id = 0


def load_data(file_path, headings, delim=","):
    """Load data from file."""
    df = pd.read_table(
        file_path,
        sep=delim,
        names=headings,
    )
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

    return X, y


def compute_kNN(q, k):
    global query_id
    response = requests.post(
        cloud_provider + "/compute_kNN",
        json={"query": q.tolist(), "k": k, "query_id": query_id},
    )
    if response.status_code != 200:
        print("Error computing kNN")
        return None
    return np.array(response.json().get("datapoints"))


def decrypt_data(data):
    response = requests.post(
        data_owner + "/decrypt_data", json={"datapoints": data.tolist()}
    )
    if response.status_code != 200:
        print("Error decrypting data")
        return None
    return np.array(response.json().get("datapoints"))


def encrypt_query(q):
    global query_id
    query_id += 1
    d = q.shape[0]
    # create diagonal matrix of d * d dimensions which contains random real numbers
    N = np.random.randint(1, 5, size=d)
    temp_N = N
    N = np.diag(N)
    q = q.reshape(1, d)
    q_dot = np.dot(q, N)
    q_dot = (B_1 * q_dot).reshape(d)
    # print("q_dot: ", q_dot)

    response = requests.post(
        data_owner + "/encrypt_query",
        json={"datapoints": q_dot.tolist(), "query_id": query_id},
    )
    if response.status_code != 200:
        print("Error encrypting query")
        return None

    q_cap = np.array(response.json().get("datapoints"))
    # print(q_cap.shape)
    n = q_cap.shape[0]
    ones = np.ones(n - d)
    # print(temp_N.shape)
    # print(ones.shape)
    N_dash = np.concatenate((temp_N, ones))
    N_dash = np.diag(N_dash)
    # print("N_dash: \n", N_dash)
    N_dash_inv = np.linalg.inv(N_dash)

    q_tilda_encrypted = np.dot(q_cap, N_dash_inv)

    # convert matrix q_tilda_encrypted to vector q_tilda_vector by adding all elements of cloumns in the row
    q_tilda_vector = np.sum(q_tilda_encrypted, axis=1)

    q_tilda_vector = np.array(q_tilda_vector)
    # print("q_tilda_vector: ", q_tilda_vector)
    return q_tilda_vector


def get_secure_knn(q, k):
    q_tilda_vector = encrypt_query(q)
    if q_tilda_vector is None:
        return None
    # print("Encrypted query: ", q_tilda_vector)

    neighbours = compute_kNN(q_tilda_vector, k)
    # print("Encrypted neighbours: ", neighbours)
    # print(neighbours)
    if neighbours is None:
        return None
    return decrypt_data(neighbours)


def load_and_process_data(file_path, headings, delim=","):
    df = load_data(file_path, headings, delim)
    X, y = process_data(df)
    return X


def main():
    test_file = "breast-cancer-wisconsin-partial.data"
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
    X_test = load_and_process_data(test_file, headings)

    q = X_test[0]
    k = 1

    print("Query: ", q)

    secure_knn = get_secure_knn(q, k)
    print("Secure kNN: ", secure_knn)


if __name__ == "__main__":
    main()
