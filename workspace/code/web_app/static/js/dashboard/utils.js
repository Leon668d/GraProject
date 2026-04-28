export function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

export function formatMilliseconds(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "--";
    }
    return `${Number(value).toFixed(2)} ms`;
}

export function formatValue(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "--";
    }
    return Number(value).toFixed(digits);
}

export function formatPercent(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "--";
    }
    return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function formatMapping(mapping) {
    if (!mapping || !Array.isArray(mapping.source) || !Array.isArray(mapping.mapped)) {
        return "--";
    }
    const [sx, sy] = mapping.source;
    const [mx, my] = mapping.mapped;
    return `(${formatValue(sx)}, ${formatValue(sy)}) -> (${formatValue(mx)}, ${formatValue(my)})`;
}

export function updateLink(id, url) {
    const link = document.getElementById(id);
    if (!link) {
        return;
    }
    if (url) {
        link.href = url;
        link.classList.remove("hidden");
    } else {
        link.classList.add("hidden");
    }
}
