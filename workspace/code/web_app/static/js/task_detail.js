function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function formatMilliseconds(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "--";
    }
    return `${Number(value).toFixed(2)} ms`;
}

function formatValue(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "--";
    }
    return Number(value).toFixed(digits);
}

function formatMapping(mapping) {
    if (!mapping || !Array.isArray(mapping.source) || !Array.isArray(mapping.mapped)) {
        return "--";
    }
    const [sx, sy] = mapping.source;
    const [mx, my] = mapping.mapped;
    return `(${formatValue(sx)}, ${formatValue(sy)}) -> (${formatValue(mx)}, ${formatValue(my)})`;
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    if (response.redirected) {
        window.location.href = response.url;
        return null;
    }

    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || "请求失败");
    }
    return data;
}

function renderDetailList(containerId, items) {
    const container = document.getElementById(containerId);
    container.innerHTML = items.map((item) => `
        <div class="detail-item">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
        </div>
    `).join("");
}

function setImageSource(id, url) {
    document.getElementById(id).src = `${url}?t=${Date.now()}`;
}

function renderSummary(task) {
    const container = document.getElementById("detail-summary");
    const cards = [
        { label: "任务状态", value: task.status || "--" },
        { label: "使用模型", value: task.model_name || "--" },
        { label: "总耗时", value: formatMilliseconds(task.timings.total_ms) },
        { label: "平均差分值", value: task.metrics.difference_mean === null || task.metrics.difference_mean === undefined ? "--" : formatValue(task.metrics.difference_mean) },
    ];
    container.innerHTML = cards.map((card) => `
        <article class="summary-card">
            <span>${escapeHtml(card.label)}</span>
            <strong>${escapeHtml(card.value)}</strong>
        </article>
    `).join("");
}

function renderBadges(task) {
    const container = document.getElementById("detail-badges");
    const badges = [
        task.status || "--",
        task.model_name || "--",
        task.method || "--",
        task.created_at || "--",
    ];
    container.innerHTML = badges.map((badge) => `<span class="badge">${escapeHtml(badge)}</span>`).join("");
}

function renderTask(task) {
    document.getElementById("detail-title").textContent = `${task.sar_name || "--"} / ${task.optical_name || "--"}`;
    document.getElementById("detail-subtitle").textContent = `创建时间：${task.created_at || "--"}，以下为该历史配准任务的完整结果记录。`;
    document.getElementById("detail-download").href = task.preview_urls.download;

    renderBadges(task);
    renderSummary(task);
    renderDetailList("detail-timings", [
        { label: "上传耗时", value: formatMilliseconds(task.timings.upload_ms) },
        { label: "读取耗时", value: formatMilliseconds(task.timings.load_ms) },
        { label: "预处理耗时", value: formatMilliseconds(task.timings.preprocess_ms) },
        { label: "模型推理耗时", value: formatMilliseconds(task.timings.registration_ms) },
        { label: "可视化耗时", value: formatMilliseconds(task.timings.visualization_ms) },
        { label: "总耗时", value: formatMilliseconds(task.timings.total_ms) },
    ]);
    renderDetailList("detail-metrics", [
        { label: "预测类型", value: task.metrics.prediction_type || "--" },
        { label: "水平位移 (dx)", value: task.metrics.dx === null || task.metrics.dx === undefined ? "--" : `${formatValue(task.metrics.dx)} px` },
        { label: "垂直位移 (dy)", value: task.metrics.dy === null || task.metrics.dy === undefined ? "--" : `${formatValue(task.metrics.dy)} px` },
        { label: "平均差分值", value: task.metrics.difference_mean === null || task.metrics.difference_mean === undefined ? "--" : formatValue(task.metrics.difference_mean) },
        { label: "模型原始 dx", value: task.metrics.raw_dx_model === null || task.metrics.raw_dx_model === undefined ? "--" : formatValue(task.metrics.raw_dx_model) },
        { label: "模型原始 dy", value: task.metrics.raw_dy_model === null || task.metrics.raw_dy_model === undefined ? "--" : formatValue(task.metrics.raw_dy_model) },
        { label: "中心映射", value: formatMapping(task.metrics.center_mapping) },
        { label: "形变平均幅值", value: task.metrics.flow_mean_magnitude === null || task.metrics.flow_mean_magnitude === undefined ? "--" : `${formatValue(task.metrics.flow_mean_magnitude)} px` },
        { label: "形变最大幅值", value: task.metrics.flow_max_magnitude === null || task.metrics.flow_max_magnitude === undefined ? "--" : `${formatValue(task.metrics.flow_max_magnitude)} px` },
        { label: "棋盘格格子数", value: task.metrics.checkerboard_cells === null || task.metrics.checkerboard_cells === undefined ? "--" : String(task.metrics.checkerboard_cells) },
        { label: "内点数 / 匹配数", value: `${task.metrics.inliers ?? "--"} / ${task.metrics.matches_used ?? "--"}` },
    ]);

    setImageSource("detail-checkerboard", task.preview_urls.checkerboard);
    setImageSource("detail-overlay", task.preview_urls.overlay);
    setImageSource("detail-difference", task.preview_urls.difference);
    setImageSource("detail-contour", task.preview_urls.contour);

    const deformationCard = document.getElementById("detail-deformation-card");
    if (task.preview_urls.deformation) {
        deformationCard.classList.remove("hidden");
        setImageSource("detail-deformation", task.preview_urls.deformation);
    } else {
        deformationCard.classList.add("hidden");
    }
}

async function init() {
    const sessionId = document.body.dataset.sessionId;
    const data = await requestJson(`/api/history/${sessionId}`);
    renderTask(data.task);
}

init().catch((error) => {
    document.getElementById("detail-title").textContent = "历史任务详情加载失败";
    document.getElementById("detail-subtitle").textContent = error.message;
});
