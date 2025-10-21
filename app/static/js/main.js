// Purestock AI - Enhanced Interface JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const sendBtn = document.getElementById('send-btn');
    const promptInput = document.getElementById('prompt-input');
    const chatContainer = document.getElementById('chat-container');
    const modelListContainer = document.getElementById('model-list-container');
    const tokenCounter = document.getElementById('token-counter');
    const modelIndicator = document.getElementById('model-indicator');
    
    let selectedModel = null;
    let isGenerating = false;
    
    // Auto-resize textarea
    promptInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        updateTokenCount();
    });
    
    // Load models list
    loadModelsList();
    
    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);
    
    // Send message on Enter (Shift+Enter for new line)
    promptInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Example prompt buttons
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('example-btn')) {
            const prompt = e.target.dataset.prompt;
            promptInput.value = prompt;
            promptInput.focus();
            updateTokenCount();
        }
    });
    
    function updateTokenCount() {
        const text = promptInput.value;
        // Rough estimation: 1 token ≈ 4 characters
        const estimatedTokens = Math.ceil(text.length / 4);
        tokenCounter.textContent = `~${estimatedTokens} tokens`;
    }
    
    function loadModelsList() {
        fetch('/models')
            .then(response => response.json())
            .then(models => {
                if (models && models.length > 0) {
                    let modelListHtml = '';
                    
                    models.forEach((model, index) => {
                        const isSelected = index === 0;
                        if (isSelected) {
                            selectedModel = model.id;
                            modelIndicator.textContent = model.name;
                        }
                        
                        modelListHtml += `
                            <div class="model-option ${isSelected ? 'selected' : ''}" data-model-id="${model.id}" data-model-name="${model.name}">
                                ${model.name}
                            </div>
                        `;
                    });
                    
                    modelListContainer.innerHTML = modelListHtml;
                    
                    // Add click handlers
                    document.querySelectorAll('.model-option').forEach(option => {
                        option.addEventListener('click', function() {
                            document.querySelectorAll('.model-option').forEach(o => o.classList.remove('selected'));
                            this.classList.add('selected');
                            selectedModel = this.dataset.modelId;
                            modelIndicator.textContent = this.dataset.modelName;
                        });
                    });
                } else {
                    modelListContainer.innerHTML = '<p style="color: #f44336; font-size: 14px; padding: 0 12px;">No models found</p>';
                }
            })
            .catch(error => {
                console.error('Error loading models:', error);
                modelListContainer.innerHTML = '<p style="color: #f44336; font-size: 14px; padding: 0 12px;">Error loading models</p>';
            });
    }
    
    function sendMessage() {
        if (isGenerating) return;
        
        const prompt = promptInput.value.trim();
        if (!prompt) return;
        
        if (!selectedModel) {
            showError('Please select a model first');
            return;
        }
        
        // Clear welcome message if it exists
        const welcomeMessage = chatContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        // Add user message
        addMessage(prompt, 'user');
        
        // Clear input
        promptInput.value = '';
        promptInput.style.height = 'auto';
        updateTokenCount();
        
        // Disable send button
        isGenerating = true;
        sendBtn.disabled = true;
        
        // Show loading
        const loadingDiv = showLoading();
        
        // Send request to server with EXAMPLE-ONLY mode (100% accurate, dataset-based)
        fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_id: selectedModel,
                prompt: prompt,
                max_tokens: 150,
                temperature: 0.0,
                deterministic: true,
                mode: 'example-only'  // PURE DATASET MODE (no hallucination)
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server error');
            }
            return response.json();
        })
        .then(data => {
            // Remove loading
            loadingDiv.remove();
            
            // Add assistant message with quality indicators
            if (data.generated_texts && data.generated_texts.length > 0) {
                const response = data.generated_texts[0];
                addMessage(response, 'assistant', {
                    tokens: data.token_count,
                    words: data.word_count,
                    time: data.generation_time,
                    mode: data.mode,
                    quality: data.quality
                });
            } else {
                showError('No response generated');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingDiv.remove();
            showError('Error: ' + error.message);
        })
        .finally(() => {
            isGenerating = false;
            sendBtn.disabled = false;
        });
    }
    
    function addMessage(content, type, metadata) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const header = type === 'user' ? '👤 You' : '🎯 Purestock AI';
        
        // Format the content for clean display
        const formattedContent = formatContent(content);
        
        let metaInfo = '';
        if (metadata && type === 'assistant') {
            const tokens = metadata.tokens || metadata;  // Support old format
            const words = metadata.words || Math.floor(tokens * 0.75);
            const time = metadata.time || 0;
            const mode = metadata.mode || 'standard';
            const quality = metadata.quality || 'standard';
            
            // Show deterministic/factual indicator
            const modeIcon = mode === 'deterministic' ? '🎯' : '🎨';
            const modeText = mode === 'deterministic' ? 'Deterministic' : 'Creative';
            const qualityText = quality.includes('hallucination') ? 'Anti-Hallucination' : quality;
            
            metaInfo = `
                <div style="font-size: 11px; color: #8e8ea0; margin-top: 8px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
                    <span>✨ ${tokens} tokens</span>
                    <span>📝 ${words} words</span>
                    <span>⚡ ${time}s</span>
                    <span style="color: #19c37d;">${modeIcon} ${modeText}</span>
                    <span style="color: #19c37d;">🛡️ ${qualityText}</span>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div class="message-header">${header}</div>
            <div class="message-content">${formattedContent}</div>
            ${metaInfo}
        `;
        
        chatContainer.appendChild(messageDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function formatContent(text) {
        // Escape HTML first
        let formatted = escapeHtml(text);
        
        // Replace line breaks with <br> tags
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Handle code blocks if present (between triple backticks)
        formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // Handle inline code (between single backticks)
        formatted = formatted.replace(/`([^`]+)`/g, '<code style="background: #2d2d2d; padding: 2px 6px; border-radius: 3px;">$1</code>');
        
        // Make URLs clickable
        formatted = formatted.replace(
            /(https?:\/\/[^\s<]+)/g,
            '<a href="$1" target="_blank" style="color: #19c37d; text-decoration: underline;">$1</a>'
        );
        
        return formatted;
    }
    
    function showLoading() {
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading';
        loadingDiv.innerHTML = `
            <div style="color: #19c37d; font-weight: 500; font-size: 14px; margin-right: 8px;">🎯 Purestock AI</div>
            <div class="loading-dots">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
        `;
        chatContainer.appendChild(loadingDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return loadingDiv;
    }
    
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message assistant';
        errorDiv.innerHTML = `
            <div class="message-header">⚠️ Error</div>
            <div class="message-content" style="color: #f44336;">${escapeHtml(message)}</div>
        `;
        chatContainer.appendChild(errorDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});