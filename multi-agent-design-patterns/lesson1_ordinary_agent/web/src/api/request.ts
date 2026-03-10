/**
 * 应用级别的 API 封装示例
 * 基于 index.ts 的基底方法，展示如何自定义业务 API
 */
import { request, get, post, put, del, patch } from './index';
import type { RequestOptions } from './index';

// ============ 自定义方法示例 ============

// 1. 直接使用快捷方法
export const getUsers = (page: number) =>
  get<any[]>('/api/users', { params: { page } });

export const createUser = (data: { name: string; email: string }) =>
  post<any>('/api/users', data);

export const updateUser = (id: string, data: any) =>
  put<any>(`/api/users/${id}`, data);

export const deleteUser = (id: string) =>
  del<any>(`/api/users/${id}`);

// 2. 使用基底 request() 传入自定义 method
export const headCheck = (url: string) =>
  request('HEAD', url);

// 3. 自定义一个带业务逻辑的高阶方法
export const postWithAuth = <T = any>(url: string, body?: any, options?: RequestOptions) => {
  const token = localStorage.getItem('token') || '';
  return post<T>(url, body, {
    ...options,
    headers: {
      ...options?.headers,
      Authorization: `Bearer ${token}`,
    },
  });
};

// 4. 文件上传
export const uploadFile = (url: string, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return post<any>(url, formData);
};
