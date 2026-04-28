import { state } from "./state.js";
import { escapeHtml, formatMapping, formatMilliseconds, formatValue } from "./utils.js";

const RESULT_IMAGES = [
    { key: "registered_preview", title: "SAR 配准后", caption: "最终展示图：将 SAR 按估计几何关系对齐到真实光学图。", id: "registered-preview-img" },
    { key: "checkerboard", title: "棋盘格对比", caption: "交替显示真实光学图与配准后的 SAR 图，用于观察边界对齐。", id: "checkerboard-img" },
    { key: "overlay", title: "伪彩色融合", caption: "融合图用于快速观察整体结构是否接近重合。", id: "overlay-img" },
    { key: "fake_optical", title: "Fake Optical", caption: "扩散模型从 SAR 生成的伪光学图，是跨模态匹配桥梁。", id: "fake-optical-img" },
    { key: "matches", title: "LightGlue 匹配", caption: "Fake Optical 与真实光学之间的对应点匹配证据。", id: "lightglue-match-img" },
    { key: "sar_transferred_matches", title: "SAR 对应点迁移", caption: "将 Fake Optical 上的对应点迁移回 SAR 图像。", id: "sar-transfer-match-img" },
    { key: "optical_transferred_points", title: "真实光学对应点", caption: "真实光学图上的 RANSAC 内点与复核点。", id: "optical-points-img" },
    { key: "difference", title: "差分图像", caption: "用于辅助观察仍存在结构差异的位置。", id: "difference-img" },
    { key: "contour", title: "轮廓叠加", caption: "用于辅助观察边缘轮廓是否接近重合。", id: "contour-img" },
    { key: "sar_condition", title: "SAR 输入图", caption: "待配准源图，同时作为扩散生成器条件输入。", id: "sar-condition-img" },
    { key: "real_optical", title: "真实光学图", caption: "配准目标图像。", id: "real-optical-img" },
];

function renderSummaryCards(timings = {}) {
    const summary = document.getElementById("timing-summary");
    const cards = [
        { label: "上传", value: formatMilliseconds(timings.upload_ms) },
        { label: "预处理", value: formatMilliseconds(timings.preprocess_ms) },
        { label: "推理", value: formatMilliseconds(timings.registration_ms) },
        { label: "总耗时", value: formatMilliseconds(timings.total_ms) },
    ];
    summary.innerHTML = cards.map((card) => `
        <article class="summary-card">
            <span>${escapeHtml(card.label)}</span>
            <strong>${escapeHtml(card.value)}</strong>
        </article>
    `).join("");
}

export function renderDetailList(containerId, items) {
    const container = document.getElementById(containerId);
    if (!container) {
        return;
    }
    container.innerHTML = items.map((item) => `
        <div class="detail-item">
            <span>${escapeHtml(item.label)}</span>
            <strong>${escapeHtml(item.value)}</strong>
        </div>
    `).join("");
}

function ensureImagePlaceholder(image) {
    const parent = image.parentElement;
    if (!parent) {
        return null;
    }
    let placeholder = parent.querySelector(".image-placeholder");
    if (!placeholder) {
        placeholder = document.createElement("div");
        placeholder.className = "image-placeholder hidden";
        placeholder.textContent = "图像暂未生成";
        image.insertAdjacentElement("afterend", placeholder);
    }
    return placeholder;
}

function showImagePlaceholder(image, message = "图像暂未生成") {
    const card = image.closest(".image-card");
    const placeholder = ensureImagePlaceholder(image);
    image.removeAttribute("src");
    image.classList.add("image-hidden");
    card?.classList.remove("image-ready");
    card?.classList.add("image-empty");
    if (placeholder) {
        placeholder.textContent = message;
        placeholder.classList.remove("hidden");
    }
}

function showImageLoaded(image) {
    const card = image.closest(".image-card");
    const placeholder = ensureImagePlaceholder(image);
    image.classList.remove("image-hidden");
    card?.classList.add("image-ready");
    card?.classList.remove("image-empty");
    placeholder?.classList.add("hidden");
}

function imageUrlWithCacheBust(url) {
    if (!url) {
        return "";
    }
    const separator = url.includes("?") ? "&" : "?";
    return `${url}${separator}t=${Date.now()}`;
}

function renderTimings(timings = {}) {
    renderSummaryCards(timings);
    renderDetailList("timing-details", [
        { label: "上传耗时", value: formatMilliseconds(timings.upload_ms) },
        { label: "读取耗时", value: formatMilliseconds(timings.load_ms) },
        { label: "预处理耗时", value: formatMilliseconds(timings.preprocess_ms) },
        { label: "模型推理耗时", value: formatMilliseconds(timings.registration_ms) },
        { label: "可视化耗时", value: formatMilliseconds(timings.visualization_ms) },
        { label: "总耗时", value: formatMilliseconds(timings.total_ms) },
    ]);
}

function renderMetrics(metrics = {}) {
    renderDetailList("metric-details", [
        { label: "质量结论", value: metrics.registration_reliable ? "可靠候选" : "待复核" },
        { label: "诊断标签", value: Array.isArray(metrics.reliability_reasons) && metrics.reliability_reasons.length ? metrics.reliability_reasons.join(", ") : "--" },
        { label: "LightGlue 匹配数", value: metrics.match_count ?? metrics.matches_total ?? "--" },
        { label: "Correct Matches", value: metrics.correct_matches ?? metrics.inliers ?? "--" },
        { label: "内点率", value: metrics.inlier_ratio === null || metrics.inlier_ratio === undefined ? "--" : formatValue(metrics.inlier_ratio, 4) },
        { label: "RMSE", value: metrics.rmse === null || metrics.rmse === undefined ? "--" : `${formatValue(metrics.rmse, 4)} px` },
        { label: "匹配策略", value: metrics.extractor_policy ? `${metrics.extractor_policy} / ${metrics.match_preprocess || "rgb"}` : "--" },
        { label: "最终特征器", value: metrics.selected_extractor || metrics.extractor || "--" },
        { label: "级联补救", value: metrics.cascade_rescued ? "是" : "否" },
        { label: "几何估计来源", value: metrics.geometry_source || "--" },
        { label: "几何应用目标", value: metrics.geometry_applied_to || "--" },
        { label: "水平位移 (dx)", value: metrics.dx === null || metrics.dx === undefined ? "--" : `${formatValue(metrics.dx)} px` },
        { label: "垂直位移 (dy)", value: metrics.dy === null || metrics.dy === undefined ? "--" : `${formatValue(metrics.dy)} px` },
        { label: "中心映射", value: formatMapping(metrics.center_mapping) },
        { label: "平均差分值", value: metrics.difference_mean === null || metrics.difference_mean === undefined ? "--" : formatValue(metrics.difference_mean) },
        { label: "扩散步数", value: metrics.generator_steps ?? "--" },
        { label: "内点数 / 匹配数", value: `${metrics.inliers ?? "--"} / ${metrics.matches_used ?? "--"}` },
    ]);
}

function renderRegistrationQuality(metrics = {}) {
    const banner = document.getElementById("registration-quality");
    if (!banner) {
        return;
    }
    const reliable = Boolean(metrics.registration_reliable);
    const reasons = Array.isArray(metrics.reliability_reasons) && metrics.reliability_reasons.length
        ? metrics.reliability_reasons
        : [];
    const reasonChips = reasons.map((reason) => `<span>${escapeHtml(reason)}</span>`).join("");

    banner.classList.remove("hidden", "unreliable");
    if (!reliable) {
        banner.classList.add("unreliable");
    }

    banner.innerHTML = reliable
        ? `
            <strong>配准质量：可靠候选</strong>
            <p>通过质量门控。SAR warp 可以作为候选配准结果，建议结合点迁移图做最终人工复核。</p>
        `
        : `
            <strong>配准质量：待复核</strong>
            <p>质量门控建议进一步复核，当前结果适合作为边界案例和诊断展示。</p>
            <div class="reason-chip-row">${reasonChips || "<span>needs_review</span>"}</div>
        `;
}

function setResultContext(text) {
    document.getElementById("result-context").textContent = text;
}

export function setImageSource(id, url, options = {}) {
    const image = document.getElementById(id);
    if (!image) {
        return;
    }
    const card = image.closest(".image-card");
    if (!url) {
        if (options.hideWhenMissing) {
            image.removeAttribute("src");
            card?.classList.add("hidden");
            return;
        }
        card?.classList.remove("hidden");
        showImagePlaceholder(image, "图像暂未生成");
        return;
    }
    card?.classList.remove("hidden");
    showImagePlaceholder(image, "正在加载图像...");
    image.onerror = () => {
        showImagePlaceholder(image, "图像文件暂不可用");
    };
    image.onload = () => {
        showImageLoaded(image);
    };
    image.src = imageUrlWithCacheBust(url);
}

export function toggleImageCard(cardId, imageId, url) {
    const card = document.getElementById(cardId);
    if (!card) {
        return;
    }
    if (url) {
        card.classList.remove("hidden");
        setImageSource(imageId, url);
    } else {
        card.classList.add("hidden");
    }
}

function setFeaturedImage(item, url) {
    const title = document.getElementById("featured-title");
    const caption = document.getElementById("featured-caption");
    if (title) {
        title.textContent = item.title;
    }
    if (caption) {
        caption.textContent = item.caption;
    }
    setImageSource("featured-img", url);
}

function renderResultGallery(previewUrls = {}) {
    const strip = document.getElementById("result-thumbnail-strip");
    if (!strip) {
        return;
    }
    const available = RESULT_IMAGES.filter((item) => previewUrls[item.key]);
    strip.innerHTML = available.length
        ? available.map((item, index) => `
            <button class="result-thumb-button${index === 0 ? " active" : ""}" type="button" data-gallery-key="${escapeHtml(item.key)}">
                <span>${escapeHtml(item.title)}</span>
            </button>
        `).join("")
        : '<div class="image-placeholder inline-placeholder">暂无可展示图像</div>';

    if (available.length) {
        setFeaturedImage(available[0], previewUrls[available[0].key]);
    } else {
        setImageSource("featured-img", null);
    }

    strip.querySelectorAll("[data-gallery-key]").forEach((button) => {
        button.addEventListener("click", () => {
            const item = RESULT_IMAGES.find((candidate) => candidate.key === button.dataset.galleryKey);
            if (!item) {
                return;
            }
            strip.querySelectorAll(".result-thumb-button").forEach((thumb) => {
                thumb.classList.toggle("active", thumb === button);
            });
            setFeaturedImage(item, previewUrls[item.key]);
        });
    });
}

export function activateTab(tabId) {
    document.querySelectorAll(".tab-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.tabTarget === tabId);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.id === tabId);
    });
}

export function setupResultTabs() {
    document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => activateTab(button.dataset.tabTarget));
    });
}

export function showTaskInResult(task, contextText) {
    const previewUrls = task.preview_urls || {};
    document.getElementById("loading").classList.add("hidden");
    document.getElementById("demo-empty-state")?.classList.add("hidden");
    document.getElementById("results").classList.remove("hidden");

    renderResultGallery(previewUrls);
    setImageSource("checkerboard-img", previewUrls.checkerboard);
    setImageSource("overlay-img", previewUrls.overlay);
    setImageSource("difference-img", previewUrls.difference);
    setImageSource("contour-img", previewUrls.contour);
    toggleImageCard("deformation-card", "deformation-img", previewUrls.deformation);
    toggleImageCard("fake-optical-card", "fake-optical-img", previewUrls.fake_optical);
    toggleImageCard("lightglue-match-card", "lightglue-match-img", previewUrls.matches);
    toggleImageCard("sar-transfer-match-card", "sar-transfer-match-img", previewUrls.sar_transferred_matches);
    toggleImageCard("optical-points-card", "optical-points-img", previewUrls.optical_transferred_points);
    toggleImageCard("registered-preview-card", "registered-preview-img", previewUrls.registered_preview);
    toggleImageCard("sar-condition-card", "sar-condition-img", previewUrls.sar_condition);
    toggleImageCard("real-optical-card", "real-optical-img", previewUrls.real_optical);

    state.registeredDownloadUrl = previewUrls.download;
    renderTimings(task.timings || {});
    renderMetrics(task.metrics || {});
    renderRegistrationQuality(task.metrics || {});

    const context = task.model_name ? `${contextText} 当前模型：${task.model_name}` : contextText;
    setResultContext(context);
    document.getElementById("current-model-status").textContent = task.model_name || "ACD_LCM_ADV:model.safetensors";
    activateTab("tab-overview");
}
