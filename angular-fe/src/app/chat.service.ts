// // import { Injectable } from '@angular/core';
// // import { HttpClient } from '@angular/common/http';
// // import { Observable } from 'rxjs';

// // export interface ChatResponse {
// //   answer: string;
// //   sources: Array<{
// //     page: string;
// //     content: string;
// //   }>;
// //   session_id: string;
// // }

// // @Injectable({
// //   providedIn: 'root'
// // })
// // export class ChatService {
// //   private apiUrl = 'https://rag-system-534297186371.us-central1.run.app';

// //   constructor(private http: HttpClient) {}

// //   chat(question: string, sessionId: string): Observable<ChatResponse> {
// //     return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, {
// //       question,
// //       session_id: sessionId
// //     });
// //   }
// // }



// import { Injectable } from '@angular/core';
// import { HttpClient } from '@angular/common/http';
// import { Observable } from 'rxjs';

// export interface ChatResponse {
//   answer: string;
//   sources: Array<{
//     page: string;
//     content: string;
//   }>;
//   session_id: string;
// }

// @Injectable({
//   providedIn: 'root'
// })
// export class ChatService {
//   private apiUrl = 'https://rag-system-534297186371.us-central1.run.app';

//   constructor(private http: HttpClient) {}

//   chat(question: string, sessionId: string): Observable<ChatResponse> {
//     return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, {
//       question,
//       session_id: sessionId
//     });
//   }
// }



// import { Injectable } from '@angular/core';
// import { HttpClient } from '@angular/common/http';
// import { Observable } from 'rxjs';

// export interface ChatResponse {
//   answer: string;
//   sources: Array<{
//     page: string;
//     content: string;
//   }>;
//   session_id: string;
// }

// @Injectable({
//   providedIn: 'root'
// })
// export class ChatService {
//   private apiUrl = 'https://rag-system-534297186371.us-central1.run.app';

//   constructor(private http: HttpClient) {}

//   chat(question: string, sessionId: string): Observable<ChatResponse> {
//     return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, {
//       question,
//       session_id: sessionId
//     });
//   }
// }


// import { Injectable } from '@angular/core';
// import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
// import { Observable, throwError } from 'rxjs';
// import { catchError, tap } from 'rxjs/operators';

// export interface ChatResponse {
//   answer: string;
//   sources: Array<{
//     page: string;
//     content: string;
//   }>;
//   session_id: string;
// }

// @Injectable({
//   providedIn: 'root'
// })
// export class ChatService {
//   private apiUrl = 'https://rag-system-534297186371.us-central1.run.app';

//   constructor(private http: HttpClient) {}

//   chat(question: string, sessionId: string): Observable<ChatResponse> {
//     const headers = new HttpHeaders({
//       'Content-Type': 'application/json'
//     });

//     const payload = {
//       question,
//       session_id: sessionId
//     };

//     console.log('Sending request:', {
//       url: `${this.apiUrl}/chat`,
//       payload,
//       headers: headers.keys()
//     });

//     return this.http.post<ChatResponse>(`${this.apiUrl}/chat`, payload, { headers })
//       .pipe(
//         tap(response => console.log('Received response:', response)),
//         catchError(this.handleError)
//       );
//   }

//   private handleError(error: HttpErrorResponse) {
//     console.error('API Error:', error);
//     let errorMessage = 'An error occurred';
    
//     if (error.error instanceof ErrorEvent) {
//       // Client-side error
//       errorMessage = `Error: ${error.error.message}`;
//     } else {
//       // Server-side error
//       errorMessage = `Error Code: ${error.status}\nMessage: ${error.message}`;
//       if (error.error) {
//         errorMessage += `\nDetails: ${JSON.stringify(error.error)}`;
//       }
//     }
    
//     return throwError(() => new Error(errorMessage));
//   }
// }


// import { Injectable } from '@angular/core';
// import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http';
// import { Observable, throwError } from 'rxjs';
// import { catchError, retry } from 'rxjs/operators';

// export interface ChatResponse {
//   answer: string;
//   sources: Array<{
//     page: string;
//     content: string;
//   }>;
//   session_id: string;
// }

// @Injectable({
//   providedIn: 'root'
// })
// export class ChatService {
//   // Using the exact URL that works in Postman
//   private apiUrl = 'https://rag-system-534297186371.us-central1.run.app';

//   constructor(private http: HttpClient) {}

//   chat(question: string, sessionId: string): Observable<ChatResponse> {
//     // Match the exact headers from Postman
//     const headers = new HttpHeaders()
//       .set('Content-Type', 'application/json')
//       .set('Accept', 'application/json');

//     // Match the exact request body format from Postman
//     const payload = {
//       question: question,
//       session_id: sessionId
//     };

//     console.log('Sending request:', {
//       url: `${this.apiUrl}/chat`,
//       headers: headers.keys(),
//       payload
//     });

//     return this.http.post<ChatResponse>(
//       `${this.apiUrl}/chat`,
//       payload,
//       { headers }
//     ).pipe(
//       retry(1),
//       catchError(this.handleError)
//     );
//   }

//   private handleError(error: HttpErrorResponse) {
//     console.error('Error details:', error);

//     if (error.status === 0) {
//       // Network error
//       console.error('An error occurred:', error.error);
//       return throwError(() => new Error('Network error. Please check your connection.'));
//     }

//     // Server error
//     const errorMessage = error.error?.message || error.message || 'Unknown error occurred';
//     return throwError(() => new Error(errorMessage));
//   }
// }



// 



// import { Injectable } from '@angular/core';
// import { HttpClient, HttpErrorResponse } from '@angular/common/http';
// import { Observable, throwError } from 'rxjs';
// import { catchError, retry } from 'rxjs/operators';

// export interface ChatResponse {
//   answer: string;
//   sources: Array<{
//     page: string;
//     content: string;
//   }>;
//   session_id: string;
// }

// @Injectable({
//   providedIn: 'root'
// })
// export class ChatService {
//   // Use the working URL from Postman
//   private apiUrl = 'https://rag-system-ouzti63wea-uc.a.run.app';

//   constructor(private http: HttpClient) {}

//   chat(question: string, sessionId: string): Observable<ChatResponse> {
//     const url = `${this.apiUrl}/chat`;
    
//     return this.http.post<ChatResponse>(
//       url,
//       { question, session_id: sessionId },
//       {
//         headers: {
//           'Content-Type': 'application/json',
//           'Accept': 'application/json',
//         },
//         responseType: 'json',
//         observe: 'body',
//         withCredentials: false
//       }
//     ).pipe(
//       retry(1),
//       catchError(this.handleError)
//     );
//   }

//   private handleError(error: HttpErrorResponse) {
//     console.error('API Error:', error);
    
//     if (error.status === 0) {
//       return throwError(() => new Error('Network error. Please check your connection.'));
//     }
//     if (error.status === 405) {
//       return throwError(() => new Error('Method not allowed. Please check the API endpoint.'));
//     }
//     if (error.status === 429) {
//       return throwError(() => new Error('Too many requests. Please try again later.'));
//     }
    
//     return throwError(() => 
//       new Error(error.error?.message || 'An unknown error occurred')
//     );
//   }
// }


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