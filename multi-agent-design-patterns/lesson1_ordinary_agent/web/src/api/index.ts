/**
 * 基底请求方法 — 基于原生 Fetch
 * 设计理念：一个核心函数 request()，method 作为参数传入；
 * 同时支持基于它自定义出 get / post / put / delete 等快捷方法。
 */

// ============ 类型定义 ============

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH' | 'HEAD' | 'OPTIONS' | string;

export interface RequestOptions {
  params?: Record<string, any>;       // URL 查询参数
  body?: any;                         // 请求体（POST/PUT/PATCH）
  headers?: Record<string, string>;   // 自定义请求头
  timeout?: number;                   // 超时时间 (ms)
  signal?: AbortSignal;               // 外部传入的中断信号
  // 允许透传原生 fetch 的其他选项
  [key: string]: any;
}

// ============ 核心基底函数 ============

/**
 * 基底请求方法
 * @param method  - HTTP 方法 ('GET' | 'POST' | ...)
 * @param url     - 请求地址
 * @param options - 请求配置
 * @returns Promise<T>
 *
 * 用法：
 *   request('GET', '/api/users', { params: { page: 1 } })
 *   request('POST', '/api/users', { body: { name: 'test' } })
 *   request('DELETE', '/api/users/1')
 */
export async function request<T = any>(
  method: HttpMethod,
  url: string,
  options: RequestOptions = {},
): Promise<T> {
  const {
    params,
    body,
    headers: customHeaders = {},
    timeout = 10000,
    signal: externalSignal,
    ...restOptions
  } = options;

  // 1. 拼接 Query 参数
  let fullUrl = url;
  if (params) {
    const qs = new URLSearchParams(
      Object.entries(params).reduce((acc, [k, v]) => {
        if (v !== undefined && v !== null) acc[k] = String(v);
        return acc;
      }, {} as Record<string, string>),
    ).toString();
    if (qs) fullUrl += (fullUrl.includes('?') ? '&' : '?') + qs;
  }

  // 2. 构建 Headers
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...customHeaders,
  };

  // 3. 构建 Body（仅非 GET/HEAD 时处理）
  let fetchBody: BodyInit | undefined;
  if (body !== undefined && method !== 'GET' && method !== 'HEAD') {
    if (body instanceof FormData || body instanceof Blob || typeof body === 'string') {
      fetchBody = body;
      // FormData 需要浏览器自动设置 Content-Type（含 boundary）
      if (body instanceof FormData) delete headers['Content-Type'];
    } else {
      fetchBody = JSON.stringify(body);
    }
  }

  // 4. 超时控制
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  // 如果外部也传了 signal，任何一个 abort 都生效
  if (externalSignal) {
    externalSignal.addEventListener('abort', () => controller.abort());
  }

  try {
    const response = await fetch(fullUrl, {
      method,
      headers,
      body: fetchBody,
      signal: controller.signal,
      ...restOptions,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    // 根据 Content-Type 自动解析
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return (await response.json()) as T;
    }
    return (await response.text()) as unknown as T;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`请求超时 (${timeout}ms): ${method} ${fullUrl}`);
    }
    throw error;
  }
}

// ============ 内置快捷方法 ============

export const get = <T = any>(url: string, options?: RequestOptions) =>
  request<T>('GET', url, options);

export const post = <T = any>(url: string, body?: any, options?: RequestOptions) =>
  request<T>('POST', url, { ...options, body });

export const put = <T = any>(url: string, body?: any, options?: RequestOptions) =>
  request<T>('PUT', url, { ...options, body });

export const del = <T = any>(url: string, options?: RequestOptions) =>
  request<T>('DELETE', url, options);

export const patch = <T = any>(url: string, body?: any, options?: RequestOptions) =>
  request<T>('PATCH', url, { ...options, body });
