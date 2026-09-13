// 聊天页面逻辑：用户注册/登录 + Redis 聊天记录回显 + 发消息渲染
const box = document.getElementById("chat-box");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");

const authPanel = document.getElementById("auth-panel");
const authTitle = document.getElementById("auth-title");
const authMsg = document.getElementById("auth-msg");
const authUsername = document.getElementById("auth-username");
const authPassword = document.getElementById("auth-password");
const authSubmit = document.getElementById("auth-submit");
const tabLogin = document.getElementById("tab-login");
const tabRegister = document.getElementById("tab-register");
const userBar = document.getElementById("user-bar");
const userName = document.getElementById("user-name");
const logoutBtn = document.getElementById("logout-btn");

// 登录态存在浏览器本地：token 用于请求，username 用于显示
let token = localStorage.getItem("cs_token") || "";
let username = localStorage.getItem("cs_username") || "";
let sessionId = localStorage.getItem("session_id");
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("session_id", sessionId);
}
let authMode = "login"; // login | register

function appendBubble(role, text) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  // 用 textContent 而不是 innerHTML：防止用户输入被当成 HTML 执行（XSS）
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

function showAuth(message = "") {
  authPanel.hidden = false;
  userBar.hidden = true;
  authMsg.textContent = message;
  authUsername.focus();
}

function showChat() {
  authPanel.hidden = true;
  userBar.hidden = false;
  userName.textContent = username;
  input.focus();
}

function switchMode(mode) {
  authMode = mode;
  const isLogin = mode === "login";
  tabLogin.classList.toggle("active", isLogin);
  tabRegister.classList.toggle("active", !isLogin);
  authTitle.textContent = isLogin ? "登录后继续对话" : "注册新账号";
  authSubmit.textContent = isLogin ? "登录" : "注册并登录";
  authMsg.textContent = "";
}

async function submitAuth() {
  const u = authUsername.value.trim();
  const p = authPassword.value;
  if (u.length < 2 || p.length < 6) {
    authMsg.textContent = "用户名至少 2 位，密码至少 6 位";
    return;
  }
  authSubmit.disabled = true;
  try {
    const res = await fetch(`/api/${authMode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: u, password: p }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "请求失败");
    token = data.token;
    username = data.username;
    localStorage.setItem("cs_token", token);
    localStorage.setItem("cs_username", username);
    box.innerHTML = "";
    showChat();
    await loadHistory();
  } catch (err) {
    authMsg.textContent = err.message;
  } finally {
    authSubmit.disabled = false;
  }
}

async function loadHistory() {
  try {
    const res = await fetch(`/api/history?token=${encodeURIComponent(token)}`);
    if (res.status === 401) {
      clearLogin();
      showAuth("登录已过期，请重新登录");
      return;
    }
    const data = await res.json();
    (data.messages || []).forEach((m) => appendBubble(m.role, m.content));
  } catch (err) {
    authMsg.textContent = "历史记录加载失败：" + err.message;
  }
}

function clearLogin() {
  token = "";
  username = "";
  localStorage.removeItem("cs_token");
  localStorage.removeItem("cs_username");
}

async function logout() {
  try {
    await fetch(`/api/logout?token=${encodeURIComponent(token)}`, { method: "POST" });
  } catch (err) {
    /* 退出失败不阻塞本地清理 */
  }
  clearLogin();
  box.innerHTML = "";
  showAuth("已退出登录");
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  if (!token) {
    showAuth("请先登录");
    return;
  }

  appendBubble("user", text);
  input.value = "";
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId, token: token }),
    });
    const data = await res.json();
    if (res.status === 401) {
      clearLogin();
      showAuth("登录已过期，请重新登录");
      return;
    }
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

authSubmit.addEventListener("click", submitAuth);
authPassword.addEventListener("keydown", (e) => {
  if (e.key === "Enter") submitAuth();
});
tabLogin.addEventListener("click", () => switchMode("login"));
tabRegister.addEventListener("click", () => switchMode("register"));
logoutBtn.addEventListener("click", logout);
sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});

// 启动：有令牌就尝试恢复会话并回显历史
(async function init() {
  switchMode("login");
  if (token) {
    showChat();
    await loadHistory();
  } else {
    showAuth();
  }
})();

