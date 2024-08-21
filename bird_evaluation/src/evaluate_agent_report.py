import sys
import json
import argparse
import sqlite3
import multiprocessing as mp
from traceback import print_exc
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm


def execute_sql(predicted_sql, ground_truth, db_path):
    conn = sqlite3.connect(db_path)
    # Connect to the database
    cursor = conn.cursor()
    cursor.execute(predicted_sql)
    predicted_res = cursor.fetchall()
    try:
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        if not ground_truth_res:
            print("Warning: Empty ground truth SQL:", ground_truth, file=sys.stderr)
    except Exception:
        print("Error in ground truth SQL:", ground_truth, file=sys.stderr)
        print_exc()
        return 0
    return int(set(predicted_res) == set(ground_truth_res))


def execute_model(predicted_sql, ground_truth, db_place, idx, meta_time_out):
    try:
        res = func_timeout(meta_time_out, execute_sql, args=(predicted_sql, ground_truth, db_place))
    except KeyboardInterrupt:
        sys.exit(0)
    except FunctionTimedOut:
        res = 0
    except Exception:
        res = 0
    return {"sql_idx": idx, "res": res}


def package_sqls(report_path, db_root_path, mode="gpt"):
    clean_sqls = []
    db_path_list = []
    with open(report_path, "r") as f:
        sql_data = json.load(f)

    if mode == "gpt":
        for x in sql_data:
            sql = x["predicted_sql"]
            db_name = x["db_id"]
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + "/" + db_name + "/" + db_name + ".sqlite")

    elif mode == "gt":
        for x in sql_data:
            sql = x["SQL"]
            db_name = x["db_id"]
            clean_sqls.append(sql)
            db_path_list.append(db_root_path + "/" + db_name + "/" + db_name + ".sqlite")

    return clean_sqls, db_path_list


def run_sqls_parallel(sqls, db_places, num_cpus, meta_time_out):
    results = []
    with mp.Pool(processes=num_cpus) as pool, tqdm(
        total=len(sqls), desc="Executing SQL pairs"
    ) as pbar:
        for i, sql_pair in enumerate(sqls):
            predicted_sql, ground_truth = sql_pair
            result = pool.apply_async(
                execute_model,
                args=(predicted_sql, ground_truth, db_places[i], i, meta_time_out),
                callback=lambda _: pbar.update(1),
            )
            results.append(result)

        return [result.get() for result in results]


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--report", type=str, required=True, default="")
    args_parser.add_argument("--db-path", type=str, required=True, default="")
    args_parser.add_argument("--mode", type=str, required=True, default="gpt")
    args_parser.add_argument("--cpus", type=int, default=None)
    args_parser.add_argument("--timeout", type=float, default=30.0)
    args = args_parser.parse_args()

    pred_queries, db_paths = package_sqls(args.report, args.db_path, mode=args.mode)
    gt_queries, _ = package_sqls(args.report, args.db_path, mode="gt")

    query_pairs = list(zip(pred_queries, gt_queries))
    exec_result = run_sqls_parallel(
        query_pairs, db_places=db_paths, num_cpus=args.cpus, meta_time_out=args.timeout
    )
    num_queries = len(exec_result)
    results = [res["res"] for res in exec_result]
    all_acc = sum(results) / num_queries
    print("Questions:", num_queries)
    print("Accuracy:", all_acc * 100)
