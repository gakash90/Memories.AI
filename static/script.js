document.addEventListener('DOMContentLoaded', () => {
    // Authentication Check
    function checkAuthentication() {
        fetch('/api/user/current')
            .then(response => {
                if (!response.ok) {
                    window.location.href = '/login';
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    document.querySelector('.profile-pic').textContent = 
                        data.user.username.substring(0, 2).toUpperCase();
                }
            })
            .catch(() => {
                window.location.href = '/login';
            });
    }
    checkAuthentication();

    // DOM Elements
    const createCollectionBtn = document.getElementById('create-collection-btn');
    const createCollectionModal = document.getElementById('create-collection-modal');
    const cancelCollectionBtn = document.getElementById('cancel-collection-btn');
    const submitCollectionBtn = document.getElementById('submit-collection-btn');
    const collectionNameInput = document.getElementById('collection-name');
    const collectionDescInput = document.getElementById('collection-description');
    const collectionsList = document.getElementById('collections-list');
    const uploadBtn = document.getElementById('upload-btn');
    const uploadModal = document.getElementById('upload-modal');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileList = document.getElementById('file-list');
    const cancelUploadBtn = document.getElementById('cancel-upload-btn');
    const submitUploadBtn = document.getElementById('submit-upload-btn');
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toast-message');
    const memoriesGrid = document.getElementById('memories-grid');
    const pageTitle = document.querySelector('.page-title');

    // Utility Functions
    const showModal = (modal) => {
        modal.style.display = 'flex';
        setTimeout(() => modal.classList.add('active'), 10);
    };

    const hideModal = (modal) => {
        modal.classList.remove('active');
        setTimeout(() => modal.style.display = 'none', 300);
    };

    const showToast = (message) => {
        toastMessage.textContent = message;
        toast.classList.add('show');
        setTimeout(() => {
            toast.classList.remove('show');
        }, 3000);
    };

    // Collection Management
    function loadCollections() {
        fetch('/api/collections')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    collectionsList.innerHTML = ''; // Clear existing collections
                    data.collections.forEach(collection => {
                        const li = document.createElement('li');
                        li.innerHTML = `
                            <a href="#" class="collection-item" data-id="${collection.id}">
                                <span class="collection-icon">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                                    </svg>
                                </span>
                                ${collection.name}
                            </a>
                        `;
                        collectionsList.appendChild(li);
                    });
                }
            })
            .catch(error => {
                console.error('Error loading collections:', error);
                showToast('Failed to load collections');
            });
    }
    loadCollections();

    // Create Collection
    createCollectionBtn.addEventListener('click', () => {
        showModal(createCollectionModal);
        collectionNameInput.focus();
    });

    cancelCollectionBtn.addEventListener('click', () => {
        hideModal(createCollectionModal);
        collectionNameInput.value = '';
        collectionDescInput.value = '';
    });

    submitCollectionBtn.addEventListener('click', () => {
        const name = collectionNameInput.value.trim();
        const description = collectionDescInput.value.trim();

        if (name) {
            fetch('/api/collections', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name, description })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    hideModal(createCollectionModal);
                    showToast('Collection created successfully!');
                    loadCollections(); // Refresh collections list
                    collectionNameInput.value = '';
                    collectionDescInput.value = '';
                } else {
                    showToast(data.error || 'Failed to create collection');
                }
            })
            .catch(error => {
                console.error('Error creating collection:', error);
                showToast('Failed to create collection');
            });
        } else {
            showToast('Please enter a collection name.');
        }
    });

    // File Upload
    let selectedCollection = null;
    let selectedFiles = [];

    uploadBtn.addEventListener('click', () => {
        // Fetch collections for upload
        fetch('/api/collections')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.collections.length > 0) {
                    showModal(uploadModal);
                } else {
                    showToast('Please create a collection first');
                }
            })
            .catch(error => {
                console.error('Error fetching collections:', error);
                showToast('Failed to load collections');
            });
    });

    cancelUploadBtn.addEventListener('click', () => {
        hideModal(uploadModal);
        fileList.innerHTML = '';
        fileInput.value = '';
        selectedFiles = [];
    });

    uploadArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        selectedFiles = Array.from(e.target.files);
        updateFileList();
    });

    function updateFileList() {
        fileList.innerHTML = '';
        selectedFiles.forEach((file, index) => {
            const li = document.createElement('li');
            li.className = 'file-item';
            li.innerHTML = `
                <span class="file-item-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                </span>
                <span class="file-item-name">${file.name}</span>
                <span class="file-item-remove" data-index="${index}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </span>
            `;
            fileList.appendChild(li);

            // Remove file functionality
            li.querySelector('.file-item-remove').addEventListener('click', () => {
                selectedFiles.splice(index, 1);
                updateFileList();
            });
        });
    }

    submitUploadBtn.addEventListener('click', () => {
        if (selectedFiles.length === 0) {
            showToast('Please select files to upload');
            return;
        }

        const activeCollection = document.querySelector('.collection-item.active');
        if (!activeCollection) {
            showToast('Please select a collection to upload to');
            return;
        }

        const collectionId = activeCollection.dataset.id;

        selectedFiles.forEach(file => {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('title', file.name);

            fetch(`/api/collections/${collectionId}/memories`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showToast(`${file.name} uploaded successfully!`);
                } else {
                    showToast(`Failed to upload ${file.name}: ${data.error}`);
                }
            })
            .catch(error => {
                console.error('Upload error:', error);
                showToast(`Error uploading ${file.name}`);
            });
        });

        hideModal(uploadModal);
        fileList.innerHTML = '';
        fileInput.value = '';
        selectedFiles = [];
    });

    // Logout
    document.getElementById('logout-btn').addEventListener('click', () => {
        fetch('/logout', { method: 'GET' })
            .then(() => {
                window.location.href = '/login';
            })
            .catch(error => {
                console.error('Logout error:', error);
                showToast('Failed to logout');
            });
    });

    // Chat Functionality (Placeholder - will be more complex)
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const chatMessagesContainer = document.getElementById('chat-messages-container');

    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        const activeCollection = document.querySelector('.collection-item.active');
        if (!activeCollection) {
            showToast('Please select a collection to chat with');
            return;
        }

        const collectionId = activeCollection.dataset.id;

        // First, create a chat session
        fetch(`/api/collections/${collectionId}/chat`, { method: 'POST' })
            .then(response => response.json())
            .then(chatData => {
                if (chatData.success) {
                    const chatId = chatData.chat.id;
                    
                    // Then send the query
                    return fetch(`/api/collections/${collectionId}/chat/${chatId}/query`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: message })
                    });
                }
                throw new Error('Failed to create chat session');
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Add user message
                    const userMessageEl = document.createElement('div');
                    userMessageEl.className = 'chat-message user';
                    userMessageEl.innerHTML = `
                        <div class="chat-avatar">JS</div>
                        <div class="chat-bubble">
                            <div class="chat-bubble-content">${message}</div>
                            <div class="chat-bubble-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                        </div>
                    `;
                    chatMessagesContainer.insertBefore(userMessageEl, chatMessagesContainer.querySelector('.typing-indicator'));

                    // Add AI response
                    const aiMessageEl = document.createElement('div');
                    aiMessageEl.className = 'chat-message';
                    aiMessageEl.innerHTML = `
                        <div class="chat-avatar">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="12" cy="12" r="10"></circle>
                                <path d="M12 6v6l4 2"></path>
                            </svg>
                        </div>
                        <div class="chat-bubble">
                            <div class="chat-bubble-content">${data.response}</div>
                            <div class="chat-bubble-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                        </div>
                    `;
                    chatMessagesContainer.insertBefore(aiMessageEl, chatMessagesContainer.querySelector('.typing-indicator'));

                    chatInput.value = '';
                    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
                }
            })
            .catch(error => {
                console.error('Chat error:', error);
                showToast('Failed to send message');
            });
    }

    chatSendBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
});