import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  template: `
    <div class="w-full h-screen bg-black flex flex-col">
      <!-- Welcome Text -->
      <div class="py-8 text-center">
        <h1 class="text-4xl font-light">
          <span class="text-blue-400">Hello,</span>
          <span class="text-pink-400 ml-2">GDGers</span>
        </h1>
      </div>
      
      <!-- Flex spacer -->
      <div class="flex-grow"></div>
      
      <!-- Input Container -->
      <div class="w-full max-w-3xl mx-auto px-4 pb-4">
        <div class="relative bg-neutral-800 rounded-2xl">
          <input
            type="text"
            [(ngModel)]="userInput"
            (keyup.enter)="sendMessage()"
            placeholder="Ask GDG Cape Town"
            class="w-full bg-transparent text-white px-6 py-4 outline-none placeholder:text-neutral-400"
          />
          <div class="absolute right-2 top-1/2 -translate-y-1/2 flex gap-2">
            <button 
              (click)="attachFile()"
              class="p-2 hover:bg-neutral-700 rounded-full transition-colors"
            >
              <svg class="w-5 h-5 text-neutral-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                <circle cx="8.5" cy="8.5" r="1.5"/>
                <polyline points="21 15 16 10 5 21"/>
              </svg>
            </button>
            <button 
              (click)="startVoice()"
              class="p-2 hover:bg-neutral-700 rounded-full transition-colors"
            >
              <svg class="w-5 h-5 text-neutral-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
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
  `]
})
export class AppComponent {
  userInput = '';

  sendMessage() {
    if (this.userInput.trim()) {
      console.log('Sending:', this.userInput);
      this.userInput = '';
    }
  }

  attachFile() {
    console.log('Attaching file...');
  }

  startVoice() {
    console.log('Starting voice input...');
  }
}