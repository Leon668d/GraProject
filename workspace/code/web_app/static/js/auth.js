const authState = {
    mode: "login",
};

function setAuthMode(mode) {
    authState.mode = mode;
    const loginTab = document.getElementById("login-tab");
    const registerTab = document.getElementById("register-tab");
    const hint = document.getElementById("auth-hint");
    const submit = document.getElementById("auth-submit");
    const message = document.getElementById("auth-message");

    loginTab.classList.toggle("active", mode === "login");
    registerTab.classList.toggle("active", mode === "register");
    submit.textContent = mode === "login" ? "登录并进入系统" : "注册并进入系统";
    hint.textContent = mode === "login"
        ? "输入现有账号后登录，或切换到注册创建新账号。"
        : "新账号会在注册成功后自动登录，并直接进入控制台。";
    message.className = "auth-message hidden";
    message.textContent = "";
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || "请求失败");
    }
    return data;
}

function showAuthMessage(text, type) {
    const message = document.getElementById("auth-message");
    message.textContent = text;
    message.className = `auth-message ${type}`;
}

function getAuthPayload() {
    return {
        username: document.getElementById("username-input").value.trim(),
        password: document.getElementById("password-input").value,
    };
}

async function submitAuth() {
    const payload = getAuthPayload();
    const endpoint = authState.mode === "login" ? "/api/auth/login" : "/api/auth/register";

    try {
        const data = await requestJson(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        showAuthMessage(data.message || "操作成功，正在进入系统。", "success");
        setTimeout(() => {
            window.location.href = "/dashboard";
        }, 600);
    } catch (error) {
        showAuthMessage(error.message, "error");
    }
}

async function checkExistingSession() {
    try {
        const data = await requestJson("/api/auth/status");
        if (data.logged_in) {
            window.location.href = "/dashboard";
        }
    } catch (error) {
        console.error("Auth status check failed:", error);
    }
}

document.getElementById("login-tab").addEventListener("click", () => setAuthMode("login"));
document.getElementById("register-tab").addEventListener("click", () => setAuthMode("register"));
document.getElementById("auth-submit").addEventListener("click", submitAuth);
document.getElementById("password-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        submitAuth();
    }
});

setAuthMode("login");
checkExistingSession();
