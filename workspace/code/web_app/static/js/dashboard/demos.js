import { requestJson } from "./api.js";
import { showTaskInResult } from "./render.js";
import { escapeHtml, formatValue } from "./utils.js";

function metricLine(metrics = {}) {
    const inliers = metrics.inliers ?? "--";
    const ratio = metrics.inlier_ratio === null || metrics.inlier_ratio === undefined
        ? "--"
        : formatValue(metrics.inlier_ratio, 3);
    const rmse = metrics.rmse === null || metrics.rmse === undefined
        ? "--"
        : `${formatValue(metrics.rmse, 3)} px`;
    return `内点 ${inliers} · 内点率 ${ratio} · RMSE ${rmse}`;
}

function sampleButton(sample) {
    return `
        <button class="demo-sample-button" type="button" data-demo-sample="${escapeHtml(sample.id)}">
            <span class="demo-sample-title">
                ${escapeHtml(sample.title || sample.id)}
                <span>${escapeHtml(sample.tag || "基准样例")}</span>
            </span>
            <span class="demo-sample-meta">
                <span>${escapeHtml(sample.season || "--")}</span>
                <span>${escapeHtml(metricLine(sample.metrics || {}))}</span>
            </span>
            <p class="demo-sample-purpose">${escapeHtml(sample.purpose || "")}</p>
        </button>
    `;
}

function renderDemoSamples(samples = []) {
    const primaryContainer = document.getElementById("demo-sample-list");
    const reviewContainer = document.getElementById("demo-review-list");
    if (!primaryContainer || !reviewContainer) {
        return;
    }

    const primary = samples.filter((sample) => sample.primary);
    const review = samples.filter((sample) => !sample.primary);

    primaryContainer.innerHTML = primary.length
        ? primary.map(sampleButton).join("")
        : '<div class="empty-block">暂无内置基准样例。</div>';
    reviewContainer.innerHTML = review.length
        ? review.map(sampleButton).join("")
        : '<div class="empty-block">暂无技术复核样例。</div>';
}

async function loadDemoSample(sampleId) {
    const data = await requestJson(`/api/demo-samples/${encodeURIComponent(sampleId)}`);
    document.querySelectorAll(".demo-sample-button").forEach((button) => {
        button.classList.toggle("active", button.dataset.demoSample === sampleId);
    });
    const demo = data.task?.demo || {};
    showTaskInResult(
        data.task,
        `已加载基准样例：${demo.title || sampleId}。${demo.purpose || ""}`
    );
}

export async function fetchDemoSamples() {
    const data = await requestJson("/api/demo-samples");
    const samples = data.samples || [];
    renderDemoSamples(samples);
    const firstPrimary = samples.find((sample) => sample.primary) || samples[0];
    if (firstPrimary) {
        await loadDemoSample(firstPrimary.id);
    }
}

export function setupDemoSampleActions() {
    ["demo-sample-list", "demo-review-list"].forEach((id) => {
        const container = document.getElementById(id);
        if (!container) {
            return;
        }
        container.addEventListener("click", (event) => {
            const button = event.target.closest("[data-demo-sample]");
            if (!button) {
                return;
            }
            loadDemoSample(button.dataset.demoSample).catch((error) => {
                alert(`样例加载未完成：${error.message}`);
            });
        });
    });
}
