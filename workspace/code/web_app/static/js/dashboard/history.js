import { requestJson } from "./api.js";
import { showTaskInResult } from "./render.js";
import { escapeHtml, formatMilliseconds, formatValue } from "./utils.js";

function displayStatus(status) {
    if (status === "failed") {
        return "未完成";
    }
    if (status === "success") {
        return "已完成";
    }
    return status || "--";
}

function renderHistorySummary(summary = {}) {
    const container = document.getElementById("history-summary");
    if (!container) {
        return;
    }

    const cards = [
        { label: "任务总数", value: summary.total_tasks ?? 0 },
        { label: "完成次数", value: summary.success_tasks ?? 0 },
        { label: "平均耗时", value: formatMilliseconds(summary.avg_total_ms) },
        { label: "最近模型", value: summary.latest_model_name || "--" },
    ];

    container.innerHTML = cards.map((card) => `
        <article class="mini-summary-card">
            <span>${escapeHtml(card.label)}</span>
            <strong>${escapeHtml(String(card.value))}</strong>
        </article>
    `).join("");
}

function renderHistory(history) {
    const historyList = document.getElementById("history-list");
    if (!history.length) {
        historyList.innerHTML = '<div class="empty-block">当前还没有历史任务，可以先加载内置样例或上传自定义图像。</div>';
        return;
    }

    historyList.innerHTML = history.map((task) => `
        <article class="history-item">
            <div class="history-head">
                <div>
                    <div class="history-name">${escapeHtml(task.sar_name || "--")} / ${escapeHtml(task.optical_name || "--")}</div>
                    <div class="history-meta">${escapeHtml(task.created_at || "--")}</div>
                </div>
                <div class="badge-row">
                    <span class="badge ${task.status === "failed" ? "failed" : ""}">${escapeHtml(displayStatus(task.status))}</span>
                    <span class="badge">${escapeHtml(task.model_name || "--")}</span>
                </div>
            </div>
            <div class="history-body">
                总耗时：${escapeHtml(formatMilliseconds(task.timings?.total_ms))}；
                内点：${escapeHtml(String(task.metrics?.inliers ?? "--"))}；
                平均差分：${escapeHtml(task.metrics?.difference_mean === null || task.metrics?.difference_mean === undefined ? "--" : formatValue(task.metrics.difference_mean))}
            </div>
            <div class="history-actions">
                <a class="history-action detail-link-button" href="/history/task/${encodeURIComponent(task.session_id)}">详情页</a>
                <button class="history-action" type="button" data-action="replay" data-session="${escapeHtml(task.session_id)}">快速回看</button>
                <button class="history-action" type="button" data-action="difference" data-url="${escapeHtml(task.preview_urls.difference)}">查看差分图</button>
                <button class="history-action" type="button" data-action="download" data-url="${escapeHtml(task.preview_urls.download)}">下载结果</button>
            </div>
        </article>
    `).join("");
}

export async function fetchHistory() {
    const data = await requestJson("/api/history?limit=10");
    renderHistorySummary(data.summary || {});
    renderHistory(data.history || []);
}

function actionLabel(action) {
    const labels = {
        upload: "上传",
        register: "配准",
        diffusion_register: "扩散配准",
        download: "下载",
        login: "登录",
        logout: "退出",
        register_user: "注册",
    };
    return labels[action] || action || "操作";
}

function buildLogDescription(log) {
    if (log.status === "failed") {
        return "状态：未完成；详情已记录在系统日志中。";
    }
    if (log.action === "upload") {
        return `上传文件：${log.files?.sar_name || "--"} / ${log.files?.optical_name || "--"}`;
    }
    if (log.action === "diffusion_register") {
        return `模型：${log.model_name || "--"}；匹配数：${log.metrics?.match_count ?? "--"}；内点：${log.metrics?.inliers ?? "--"}；RMSE：${log.metrics?.rmse ?? "--"}`;
    }
    if (log.action === "download") {
        return `下载文件：${log.filename || "--"}`;
    }
    if (["login", "logout", "register_user"].includes(log.action)) {
        return `用户：${log.username || "--"}`;
    }
    return "暂无详细信息";
}

function renderLogs(logs) {
    const logList = document.getElementById("log-list");
    if (!logs.length) {
        logList.innerHTML = '<div class="empty-block">暂无操作日志。</div>';
        return;
    }

    logList.innerHTML = logs.map((log) => `
        <article class="log-item">
            <div class="history-head">
                <div class="log-meta">${escapeHtml(log.timestamp || "--")}</div>
                <div class="badge-row">
                    <span class="badge ${log.status === "failed" ? "failed" : ""}">${escapeHtml(actionLabel(log.action))}</span>
                </div>
            </div>
            <div>${escapeHtml(buildLogDescription(log))}</div>
        </article>
    `).join("");
}

export async function fetchRecentLogs() {
    const data = await requestJson("/api/logs?limit=8");
    renderLogs(data.logs || []);
}

export async function loadHistoryTask(sessionId) {
    try {
        const data = await requestJson(`/api/history/${sessionId}`);
        showTaskInResult(data.task, `当前为历史任务 ${data.task.created_at || ""} 的回看结果。`);
    } catch (error) {
        alert(`历史任务加载未完成：${error.message}`);
    }
}
