"""Prepare or idempotently add the evidenced AI stock list to an existing Futu group.

No trade context, orders, money or account numbers. Creating a group remains a
client action because OpenD does not expose it. Run --apply only to add members.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent


def run_child(apply):
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    payload = json.loads((OUT / "data.json").read_text())
    rows = [r for r in payload["rows"] if r["kind"] == "STOCK"
            and r["business"]["tier"] in (1, 2, 3)]
    codes = [r["code"] for r in rows]
    group_names = {g["id"]: g["name"] for g in payload["groups"]}
    (OUT / "ai_basket_members.json").write_text(json.dumps(dict(
        built_at=payload["built_at"], scope="全部已取得业务证据的1/2/3级股票；未评级不自动加入。",
        members=[dict(code=r["code"], name=r["name"], tier=r["business"]["tier"],
                      groups=r["business"]["groups"], reasons=r["business"]["reasons"])
                 for r in rows]), ensure_ascii=False, indent=2))
    with (OUT / "ai_basket_members.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["证券代码", "公司", "AI关联等级", "产业群", "关联依据"])
        writer.writerows([r["code"], r["name"], r["business"]["tier"],
                          " | ".join(group_names[g] for g in r["business"]["groups"]), " | ".join(r["business"]["reasons"])] for r in rows)
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(),
        target_group="AI 篮子", requested_count=len(codes), requested_codes=codes,
        complete=False, applied=False, verified_requested_count=0)
    q = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        ret, groups = q.get_user_security_group()
        if ret != ft.RET_OK:
            result["message"] = "futud 分组读取失败：" + str(groups)
        else:
            names = list(groups.group_name)
            target = "AI 篮子" if "AI 篮子" in names else next((n for n in names if n.replace(" ", "") == "AI篮子"), None)
            result["group_exists"] = target is not None
            if not target:
                result["message"] = "futud 尚无「AI 篮子」分组；全部名单已准备，需先在牛牛客户端新建分组。"
            else:
                result["actual_group"] = target
                ret, current = q.get_user_security(target)
                if ret != ft.RET_OK:
                    result["message"] = "分组成员读取失败，未添加。"
                else:
                    before = set(current.code)
                    missing = [c for c in codes if c not in before]
                    result["existing_requested_count"] = len(set(codes) & before)
                    result["missing_count"] = len(missing)
                    attempts = []
                    if apply:
                        for offset in range(0, len(missing), 100):
                            batch = missing[offset:offset + 100]
                            ret, msg = q.modify_user_security(target, ft.ModifyUserSecurityOp.ADD, batch)
                            attempts.append(dict(codes=batch, ret=ret, message=str(msg)))
                            if ret != ft.RET_OK:
                                break
                        result["applied"] = bool(attempts)
                    result["attempts"] = attempts
                    ret, current = q.get_user_security(target)
                    verified = set(current.code) if ret == ft.RET_OK else set()
                    result["verified_requested_count"] = len(set(codes) & verified)
                    result["unconfirmed_codes"] = [c for c in codes if c not in verified]
                    result["complete"] = not result["unconfirmed_codes"]
                    result["message"] = ("全部成员已回读确认。" if result["complete"] else
                                         "名单已准备，成员尚未全部同步；保留未确认代码。")
    finally:
        q.close()
    (OUT / "ai_basket_action.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("requested_codes", "attempts", "unconfirmed_codes")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child(args.apply)
    else:
        command = [sys.executable, str(Path(__file__).resolve()), "--child"]
        if args.apply:
            command.append("--apply")
        try:
            subprocess.run(command, timeout=30, check=True)
        except subprocess.TimeoutExpired:
            raise SystemExit("OpenD 30秒预算耗尽；若执行过添加，必须重新读取成员确认结果。")
