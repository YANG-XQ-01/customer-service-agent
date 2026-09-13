"""轻量压测脚本（零额外依赖，复用 httpx）。

用法示例：
    python scripts/loadtest.py --mode health --users 10 --requests 10
    python scripts/loadtest.py --mode chat --users 3 --requests 1
    python scripts/loadtest.py --mode chat --users 5 --requests 2 --message "耳机防水吗"

说明：
- health 模式不调用大模型，适合验证并发与吞吐；
- chat 模式会真实调用模型（花钱），默认发很少请求；
- 429 单独统计（命中限流是预期行为，不算失败）。
"""
import argparse
import asyncio
import statistics
import time
import uuid

import httpx


async def one_request(client, mode, base_url, token, message):
    if mode == "health":
        return await client.get(f"{base_url}/health")
    payload = {"message": message, "token": token}
    return await client.post(f"{base_url}/api/chat", json=payload)


async def worker(client, args, token, results):
    for _ in range(args.requests):
        started = time.perf_counter()
        try:
            response = await one_request(client, args.mode, args.base_url, token, args.message)
            elapsed = (time.perf_counter() - started) * 1000
            results.append((response.status_code, elapsed))
        except Exception as exc:  # 网络异常也算一次失败
            elapsed = (time.perf_counter() - started) * 1000
            results.append((0, elapsed))
            print(f"  请求异常: {exc}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--mode", choices=["health", "chat"], default="health")
    parser.add_argument("--users", type=int, default=10, help="并发用户数")
    parser.add_argument("--requests", type=int, default=5, help="每个用户发起的请求数")
    parser.add_argument("--message", default="帮我查一下订单20260901001到哪了")
    parser.add_argument("--token", default="", help="已有登录令牌；不填则自动注册压测账号")
    args = parser.parse_args()

    async with httpx.AsyncClient(timeout=120) as client:
        token = args.token
        if args.mode == "chat" and not token:
            username = "loadtest_" + uuid.uuid4().hex[:8]
            res = await client.post(
                f"{args.base_url}/api/register",
                json={"username": username, "password": "loadtest123"},
            )
            res.raise_for_status()
            token = res.json()["token"]
            print(f"压测账号: {username}")

        total = args.users * args.requests
        print(f"开始压测: mode={args.mode} users={args.users} requests={args.requests} total={total}")
        results: list[tuple[int, float]] = []
        started = time.perf_counter()
        await asyncio.gather(
            *[worker(client, args, token, results) for _ in range(args.users)]
        )
        duration = time.perf_counter() - started

    ok = [ms for status, ms in results if 200 <= status < 300]
    limited = [ms for status, ms in results if status == 429]
    failed = [(status, ms) for status, ms in results if status == 0 or status >= 500]
    other = [status for status, _ in results if status not in (0,) and not (200 <= status < 300) and status != 429 and status < 500]
    latencies = sorted(ms for _, ms in results)

    def pct(p):
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 1)

    print("\n===== 压测结果 =====")
    print(f"总请求: {len(results)} | 耗时: {duration:.2f}s | 吞吐: {len(results)/duration:.2f} req/s")
    print(f"成功(2xx): {len(ok)} | 限流(429): {len(limited)} | 5xx/网络失败: {len(failed)} | 其它4xx: {len(other)}")
    if latencies:
        print(f"延迟(ms): p50={pct(0.50)} p95={pct(0.95)} p99={pct(0.99)} max={round(max(latencies),1)}")
        if ok:
            print(f"成功请求平均延迟: {round(statistics.mean(ok),1)} ms")
    print("\n提示：打开 " + args.base_url + "/dashboard 可看到本次压测产生的指标。")


if __name__ == "__main__":
    asyncio.run(main())

