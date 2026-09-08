// 聊天页面逻辑：发消息、渲染气泡、记住会话
const box = document.getElementById("chat-box");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

// 会话 ID：第一次打开页面时生成一次，之后一直复用（存在浏览器 localStorage 里）。
// 这样即使刷新页面，后端也能认出“还是同一个用户”，对话记忆不丢。
let sessionId = localStorage.getItem("session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("session_id", sessionId);
}

function appendBubble(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  // 用 textContent 而不是 innerHTML：防止用户输入的内容被当成 HTML 执行（XSS 攻击）
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;

  appendBubble("user", text);
  input.value = "";
  sendBtn.disabled = true; // 等待期间禁止重复发送

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "请求失败");
    appendBubble("assistant", data.reply);
    if (data.needs_human && data.handoff) {
      appendBubble(
        "system",
        "🔔 已为您转接人工客服，工单号：" +
        data.handoff.ticket_id +
        "\n转接原因：" + data.handoff.reason +
        "\n人工客服正在查看对话记录，请稍候。"
      );
    }
  } catch (err) {
    appendBubble("assistant", "出错了：" + err.message);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});
