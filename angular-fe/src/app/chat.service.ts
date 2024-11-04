
import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, retry } from 'rxjs/operators';

export interface ChatResponse {
  answer: string;
  sources: Array<{
    page: string;
    content: string;
  }>;
  session_id: string;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = '/api'; // Use the proxied path

  constructor(private http: HttpClient) {}

  chat(question: string, sessionId: string): Observable<ChatResponse> {
    const url = `${this.apiUrl}/chat`;
    
    return this.http.post<ChatResponse>(
      url,
      { question, session_id: sessionId },
      {
        headers: {
          'Content-Type': 'application/json',
        }
      }
    ).pipe(
      retry(1),
      catchError(this.handleError)
    );
  }

  private handleError(error: HttpErrorResponse) {
    console.error('API Error:', error);
    
    if (error.status === 0) {
      return throwError(() => new Error('Network error. Please check your connection.'));
    }
    
    return throwError(() => 
      new Error(error.error?.message || 'An unknown error occurred')
    );
  }
}