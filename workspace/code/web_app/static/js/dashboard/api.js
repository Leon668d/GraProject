export async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    if (response.redirected) {
        window.location.href = response.url;
        return null;
    }

    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || "请求未完成");
    }
    return data;
}
