function askQuestion() {
    let questionInput = document.getElementById("question");
    let chatBox = document.getElementById("chat-box");

    let question = questionInput.value.trim();
    if (question === "") return;

    // Display user message
    let userMessage = document.createElement("div");
    userMessage.className = "user-message";
    userMessage.innerText = question;
    chatBox.appendChild(userMessage);

    questionInput.value = "";

    // Display typing animation
    let botMessage = document.createElement("div");
    botMessage.className = "bot-message typing";
    botMessage.innerHTML = "Typing...";
    chatBox.appendChild(botMessage);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Send request to backend
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        botMessage.classList.remove("typing");
        botMessage.innerHTML = marked.parse(data.answer); // ✅ Parse Markdown to HTML
    })
    .catch(error => {
        botMessage.classList.remove("typing");
        botMessage.innerText = "Error fetching response!";
    });

    chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll
}
