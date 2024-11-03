// import { Component } from '@angular/core';
// import { CommonModule } from '@angular/common';
// import { FormsModule } from '@angular/forms';

// @Component({
//   selector: 'app-root',
//   standalone: true,
//   imports: [CommonModule, FormsModule],
//   template: `
//     <div class="w-full h-screen bg-black flex flex-col">
//       <!-- Welcome Text -->
//       <div class="py-8 text-center">
//         <h1 class="text-4xl font-light">
//           <span class="text-blue-400">Hello,</span>
//           <span class="text-pink-400 ml-2">GDGers</span>
//         </h1>
//       </div>
      
//       <!-- Flex spacer -->
//       <div class="flex-grow"></div>
      
//       <!-- Input Container -->
//       <div class="w-full max-w-3xl mx-auto px-4 pb-4">
//         <div class="relative bg-neutral-800 rounded-2xl">
//           <input
//             type="text"
//             [(ngModel)]="userInput"
//             (keyup.enter)="sendMessage()"
//             placeholder="Ask GDG Cape Town"
//             class="w-full bg-transparent text-white px-6 py-4 outline-none placeholder:text-neutral-400"
//           />
//           <div class="absolute right-2 top-1/2 -translate-y-1/2 flex gap-2">
//             <button 
//               (click)="attachFile()"
//               class="p-2 hover:bg-neutral-700 rounded-full transition-colors"
//             >
//               <svg class="w-5 h-5 text-neutral-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
//                 <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
//                 <circle cx="8.5" cy="8.5" r="1.5"/>
//                 <polyline points="21 15 16 10 5 21"/>
//               </svg>
//             </button>
//             <button 
//               (click)="startVoice()"
//               class="p-2 hover:bg-neutral-700 rounded-full transition-colors"
//             >
//               <svg class="w-5 h-5 text-neutral-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
//                 <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
//                 <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
//                 <line x1="12" y1="19" x2="12" y2="23"/>
//                 <line x1="8" y1="23" x2="16" y2="23"/>
//               </svg>
//             </button>
//           </div>
//         </div>
//       </div>
//     </div>
//   `,
//   styles: [`
//     :host {
//       display: block;
//       height: 100vh;
//     }
//   `]
// })
// export class AppComponent {
//   userInput = '';

//   sendMessage() {
//     if (this.userInput.trim()) {
//       console.log('Sending:', this.userInput);
//       this.userInput = '';
//     }
//   }

//   attachFile() {
//     console.log('Attaching file...');
//   }

//   startVoice() {
//     console.log('Starting voice input...');
//   }
// }


import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { ChatService } from './chat.service';

interface Message {
  content: string;
  isUser: boolean;
  sources?: Array<{
    page: string;
    content: string;
  }>;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  template: `
    <div class="w-full h-screen bg-black flex flex-col">
      <!-- Welcome Text -->
      <div class="py-8 text-center">
        <h1 class="text-4xl font-light">
          <span class="text-blue-400">Hello,</span>
          <span class="text-pink-400 ml-2">GDGers</span>
        </h1>
      </div>
      
      <!-- Chat Messages -->
      <div class="flex-grow overflow-y-auto px-4 pb-4" #scrollContainer>
        <div class="max-w-3xl mx-auto space-y-4">
          <div *ngFor="let message of messages" 
               [class]="message.isUser ? 'flex justify-end' : 'flex justify-start'">
            <div [class]="message.isUser ? 
                         'bg-blue-500 text-white rounded-2xl py-2 px-4 max-w-xl' : 
                         'bg-neutral-800 text-white rounded-2xl py-2 px-4 max-w-xl'">
              <p class="whitespace-pre-wrap">{{message.content}}</p>
              
              <!-- Sources -->
              <div *ngIf="message.sources && message.sources.length > 0" 
                   class="mt-2 pt-2 border-t border-neutral-700 text-sm">
                <div *ngFor="let source of message.sources" class="mt-1">
                  <div class="text-neutral-400">From page {{source.page}}:</div>
                  <div class="text-neutral-300">{{source.content}}</div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Loading indicator -->
          <div *ngIf="isLoading" class="flex justify-start">
            <div class="bg-neutral-800 text-white rounded-2xl py-2 px-4">
              <div class="flex gap-2">
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                <div class="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style="animation-delay: 0.4s"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- Input Container -->
      <div class="w-full max-w-3xl mx-auto px-4 pb-4">
        <div class="relative bg-neutral-800 rounded-2xl">
          <input
            type="text"
            [(ngModel)]="userInput"
            (keyup.enter)="sendMessage()"
            [disabled]="isLoading"
            placeholder="Ask about the Africa Developer Ecosystem Report"
            class="w-full bg-transparent text-white px-6 py-4 outline-none placeholder:text-neutral-400"
          />
          <div class="absolute right-2 top-1/2 -translate-y-1/2 flex gap-2">
            <button 
              [disabled]="isLoading || !userInput.trim()"
              (click)="sendMessage()"
              class="p-2 hover:bg-neutral-700 rounded-full transition-colors disabled:opacity-50"
            >
              <svg class="w-5 h-5 text-neutral-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 2L11 13"></path>
                <path d="M22 2L15 22L11 13L2 9L22 2z"></path>
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    :host {
      display: block;
      height: 100vh;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
      width: 8px;
    }
    
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
      background: #4a5568;
      border-radius: 4px;
    }
  `]
})
export class AppComponent {
  userInput = '';
  messages: Message[] = [];
  isLoading = false;
  sessionId = `session-${Date.now()}`;

  constructor(private chatService: ChatService) {
    // Add welcome message
    this.messages.push({
      content: "Hello! I'm your AI assistant. Ask me anything about the Africa Developer Ecosystem Report.",
      isUser: false
    });
  }

  // sendMessage() {
  //   if (!this.userInput.trim() || this.isLoading) return;

  //   // Add user message
  //   this.messages.push({
  //     content: this.userInput,
  //     isUser: true
  //   });

  //   const userQuestion = this.userInput;
  //   this.userInput = '';
  //   this.isLoading = true;

  //   // Call API
  //   this.chatService.chat(userQuestion, this.sessionId).subscribe({
  //     next: (response) => {
  //       this.messages.push({
  //         content: response.answer,
  //         isUser: false,
  //         sources: response.sources
  //       });
  //     },
  //     error: (error) => {
  //       console.error('Error:', error);
  //       this.messages.push({
  //         content: 'Sorry, I encountered an error. Please try again.',
  //         isUser: false
  //       });
  //     },
  //     complete: () => {
  //       this.isLoading = false;
  //       this.scrollToBottom();
  //     }
  //   });
  // }


  // sendMessage() {
  //   if (!this.userInput.trim() || this.isLoading) return;

  //   // Add user message
  //   this.messages.push({
  //     content: this.userInput,
  //     isUser: true
  //   });

  //   const userQuestion = this.userInput;
  //   this.userInput = '';
  //   this.isLoading = true;

  //   // Call API
  //   this.chatService.chat(userQuestion, this.sessionId).subscribe({
  //     next: (response) => {
  //       console.log('Success response:', response);
  //       this.messages.push({
  //         content: response.answer,
  //         isUser: false,
  //         sources: response.sources
  //       });
  //     },
  //     error: (error) => {
  //       console.error('Component error handler:', error);
  //       this.messages.push({
  //         content: `Error: ${error.message || 'Something went wrong. Please try again.'}`,
  //         isUser: false
  //       });
  //     },
  //     complete: () => {
  //       this.isLoading = false;
  //       this.scrollToBottom();
  //     }
  //   });
  // }

  sendMessage() {
    if (!this.userInput.trim() || this.isLoading) return;

    // Store the question
    const userQuestion = this.userInput;
    
    // Add user message
    this.messages.push({
      content: userQuestion,
      isUser: true
    });
    
    // Clear input and show loading
    this.userInput = '';
    this.isLoading = true;

    console.log('Sending chat request...');

    this.chatService.chat(userQuestion, this.sessionId).subscribe({
      next: (response) => {
        console.log('Chat response received:', response);
        this.messages.push({
          content: response.answer,
          isUser: false,
          sources: response.sources
        });
      },
      error: (error) => {
        console.error('Chat error:', error);
        let errorMessage = 'An error occurred. Please try again.';
        
        if (error.status === 0) {
          errorMessage = 'Network error. Please check your connection and try again.';
        } else if (error.error) {
          errorMessage = error.error.message || error.message || errorMessage;
        }
        
        this.messages.push({
          content: errorMessage,
          isUser: false
        });
      },
      complete: () => {
        console.log('Chat request completed');
        this.isLoading = false;
        this.scrollToBottom();
      }
    });
  }

  private scrollToBottom() {
    setTimeout(() => {
      const scrollContainer = document.querySelector('#scrollContainer');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }, 0);
  }
}