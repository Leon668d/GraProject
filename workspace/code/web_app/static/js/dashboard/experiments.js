import { requestJson } from "./api.js";
import { setImageSource } from "./render.js";
import { escapeHtml, formatPercent, formatValue, updateLink } from "./utils.js";

function renderMetricCards(containerId, cards, cardClass = "summary-card") {
    const container = document.getElementById(containerId);
    if (!container) {
        return;
    }
    container.innerHTML = cards.map((card) => `
        <article class="${cardClass}">
            <span>${escapeHtml(card.label)}</span>
            <strong>${escapeHtml(String(card.value))}</strong>
        </article>
    `).join("");
}

function renderDiffusionMetricCards(summary = {}, artifacts = {}) {
    renderMetricCards("diffusion-baseline-metrics", [
        { label: "样本数", value: summary.sample_count ?? "--" },
        { label: "平均匹配数", value: summary.avg_match_count === undefined ? "--" : formatValue(summary.avg_match_count, 2) },
        { label: "平均置信度", value: summary.avg_mean_score === undefined ? "--" : formatValue(summary.avg_mean_score, 4) },
        { label: "平均内点数", value: summary.avg_inliers === undefined ? "--" : formatValue(summary.avg_inliers, 2) },
        { label: "平均内点率", value: summary.avg_inlier_ratio === undefined ? "--" : formatValue(summary.avg_inlier_ratio, 4) },
        { label: "Fake Optical", value: artifacts.fake_optical_count ?? "--" },
    ]);
}

function renderDiffusionSampleTable(rows = []) {
    const container = document.getElementById("diffusion-sample-table");
    if (!container) {
        return;
    }
    if (!rows.length) {
        container.innerHTML = '<div class="empty-block">未找到 LightGlue 评估记录。</div>';
        return;
    }

    container.innerHTML = `
        <div class="baseline-table-head">
            <span>样本</span>
            <span>匹配数</span>
            <span>内点</span>
            <span>内点率</span>
        </div>
        ${rows.slice(0, 8).map((row) => `
            <div class="baseline-table-row">
                <span title="${escapeHtml(row.file || "--")}">${escapeHtml(row.file || "--")}</span>
                <strong>${escapeHtml(formatValue(row.match_count, 0))}</strong>
                <strong>${escapeHtml(formatValue(row.inliers, 0))}</strong>
                <strong>${escapeHtml(formatValue(row.inlier_ratio, 3))}</strong>
            </div>
        `).join("")}
    `;
}

export async function fetchDiffusionBaseline() {
    const statusPill = document.getElementById("diffusion-baseline-status");
    if (!statusPill) {
        return;
    }
    try {
        const data = await requestJson("/api/diffusion-baseline/status");
        const generator = data.generator || {};
        const artifacts = data.artifacts || {};
        const lightglue = data.lightglue || {};
        const primary = lightglue.primary;
        const primarySummary = primary ? (lightglue.candidates || {})[primary] || {} : {};

        statusPill.textContent = generator.checkpoint_exists && artifacts.metrics_csv_exists ? "Baseline 已就绪" : "Baseline 文件待补全";
        statusPill.classList.toggle("missing", !(generator.checkpoint_exists && artifacts.metrics_csv_exists));

        document.getElementById("diffusion-generator-name").textContent = generator.name || "--";
        document.getElementById("diffusion-generator-path").textContent = generator.checkpoint_path || "--";

        renderDiffusionMetricCards(primarySummary, artifacts);
        renderDiffusionSampleTable(lightglue.rows || []);

        if (artifacts.contact_sheet_url) {
            setImageSource("diffusion-contact-sheet", artifacts.contact_sheet_url);
        }
        updateLink("diffusion-download-metrics", artifacts.metrics_csv_url);
        updateLink("diffusion-download-manifest", artifacts.manifest_csv_url);
    } catch (error) {
        statusPill.textContent = "Baseline 待读取";
        statusPill.classList.add("missing");
        renderMetricCards("diffusion-baseline-metrics", [{ label: "提示", value: error.message }]);
    }
}

function renderExperimentBestCards(bestParams = {}, bestSummary = {}, contactSheet = {}) {
    const strategy = bestParams.matcher_strategy || bestParams.extractor || "cascade-superpoint+aliked-rgb";
    renderMetricCards("experiment-best-cards", [
        { label: "最终策略", value: strategy },
        { label: "可靠成功率", value: formatPercent(bestSummary.success_rate) },
        { label: "可靠样本", value: `${bestSummary.reliable_count ?? "--"} / ${bestSummary.n ?? "--"}` },
        { label: "RMSE 中位数", value: bestSummary.median_rmse_reliable === undefined ? "--" : `${formatValue(bestSummary.median_rmse_reliable, 4)} px` },
        { label: "级联救回", value: bestSummary.cascade_rescued_count ?? contactSheet.cascade_rescued_count ?? "--" },
        { label: "默认参数", value: `${bestParams.steps ?? 8} steps / ${bestParams.max_keypoints ?? 2048} kp` },
    ]);
}

function renderStrategyTable(rows = []) {
    const container = document.getElementById("experiment-strategy-table");
    if (!container) {
        return;
    }
    if (!rows.length) {
        container.innerHTML = '<div class="empty-block">未找到 summary_by_params.csv。</div>';
        return;
    }

    container.innerHTML = `
        <div class="experiment-table-head">
            <span>策略</span>
            <span>成功率</span>
            <span>RMSE</span>
            <span>内点</span>
            <span>耗时</span>
        </div>
        ${rows.map((row) => `
            <div class="experiment-table-row ${row.matcher_strategy === "cascade-superpoint+aliked-rgb" ? "best" : ""}">
                <span title="${escapeHtml(row.matcher_strategy || "--")}">${escapeHtml(row.matcher_strategy || "--")}</span>
                <strong>${escapeHtml(formatPercent(row.success_rate))}</strong>
                <strong>${escapeHtml(row.median_rmse_reliable === undefined ? "--" : formatValue(row.median_rmse_reliable, 3))}</strong>
                <strong>${escapeHtml(row.median_inliers === undefined ? "--" : formatValue(row.median_inliers, 1))}</strong>
                <strong>${escapeHtml(row.median_total_ms === undefined ? "--" : `${formatValue(row.median_total_ms, 0)} ms`)}</strong>
            </div>
        `).join("")}
    `;
}

function renderFilterSummary(summary = {}) {
    const container = document.getElementById("filter-summary-grid");
    if (!container) {
        return;
    }
    if (!summary.input_count) {
        container.innerHTML = '<div class="empty-block">未找到 filter_summary.json。</div>';
        return;
    }

    const thresholds = summary.thresholds || {};
    const reasons = summary.rejection_reasons || {};
    container.innerHTML = `
        <article><span>输入样本</span><strong>${escapeHtml(summary.input_count)}</strong></article>
        <article><span>保留样本</span><strong>${escapeHtml(summary.kept_count)}</strong></article>
        <article><span>筛出样本</span><strong>${escapeHtml(summary.rejected_count)}</strong></article>
        <article><span>保留比例</span><strong>${escapeHtml(formatPercent(summary.kept_ratio))}</strong></article>
        <article><span>边缘阈值</span><strong>${escapeHtml(thresholds.min_edge ?? "--")}</strong></article>
        <article><span>水体阈值</span><strong>${escapeHtml(thresholds.max_water ?? "--")}</strong></article>
        <article><span>低纹理</span><strong>${escapeHtml(reasons.low_texture ?? "--")}</strong></article>
        <article><span>水体 proxy</span><strong>${escapeHtml(reasons.water_like ?? "--")}</strong></article>
    `;
}

export async function fetchExperimentSummary() {
    const statusPill = document.getElementById("experiment-summary-status");
    try {
        const data = await requestJson("/api/experiment-summary");
        const artifacts = data.artifacts || {};
        statusPill.textContent = data.study?.summary_csv_exists ? "实验结果已加载" : "实验结果待补全";
        statusPill.classList.toggle("missing", !data.study?.summary_csv_exists);

        renderExperimentBestCards(data.best_params || {}, data.best_summary || {}, data.contact_sheet || {});
        renderStrategyTable(data.strategy_rows || []);
        renderFilterSummary(data.filter_summary || {});

        if (artifacts.cascade_rescued_preview_url || artifacts.cascade_rescued_url) {
            setImageSource("experiment-cascade-rescued", artifacts.cascade_rescued_preview_url || artifacts.cascade_rescued_url);
        }
        if (artifacts.failures_all_preview_url || artifacts.failures_all_url) {
            setImageSource("experiment-failures-all", artifacts.failures_all_preview_url || artifacts.failures_all_url);
        }

        updateLink("experiment-download-summary", artifacts.summary_csv_url);
        updateLink("experiment-download-all-runs", artifacts.all_runs_csv_url);
        updateLink("experiment-download-paper-table", artifacts.paper_table_url);
        updateLink("experiment-download-filtered", artifacts.filtered_pairs_url);
        updateLink("experiment-download-rejected", artifacts.rejected_pairs_url);
    } catch (error) {
        statusPill.textContent = "实验摘要待读取";
        statusPill.classList.add("missing");
        renderMetricCards("experiment-best-cards", [{ label: "提示", value: error.message }]);
    }
}
