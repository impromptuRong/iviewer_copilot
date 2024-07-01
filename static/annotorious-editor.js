var AnnotatorWidget = function(args) {
    var currentAnnotator = args.annotation ? args.annotation.bodies.find(function(b) {
        return b.purpose == 'showannotator';
    }) : null;
    var currentAnnotatorValue = currentAnnotator ? currentAnnotator.value : userId;

    var currentCreateAt = args.annotation ? args.annotation.bodies.find(function(b) {
        return b.purpose == 'showtime';
    }) : null;
    var currentCreateTimeISO = currentCreateAt ? currentCreateAt.value : null;

    var annotatorElement = document.createElement('p');
    annotatorElement.textContent = "Last annotated by: " + userIdNameMap[currentAnnotatorValue];
    if (currentCreateTimeISO != null) {
        // Convert local time
        const isoTime = new Date(currentCreateTimeISO);
        const localTime = new Date( isoTime.getTime() - isoTime.getTimezoneOffset() * 60000);
        annotatorElement.textContent += '\n' + localTime.toLocaleString();
    }
    annotatorElement.className = 'annotator';

    var container = document.createElement('div');
    container.className = 'annotator-widget';
    container.classList.add('r6o-draggable');
    container.appendChild(annotatorElement);

    return container;
}

var aiChatBox = function(args) {
    // Create container for chatbox
    var chatboxContainer = document.createElement('div');
    chatboxContainer.className = 'chatbox-widget';
    chatboxContainer.style.display = 'none'; // Hide the chatbox initially

    // Create button and button container
    var button = document.createElement('button');
    button.className = 'btn btn-sm btn-primary ai-img-caption';
    button.textContent = 'Open I-Viewer Copilot';

    var buttonContainer = document.createElement('div');
    buttonContainer.classList.add('r6o-draggable');
    buttonContainer.appendChild(button);

    // Create container to include both button and chatbox
    var container = document.createElement('div');
    container.appendChild(buttonContainer);
    container.appendChild(chatboxContainer);

    // Button event listener
    button.addEventListener('click', function () {
        var commentContainer = document.getElementsByClassName("r6o-widget")[0];
        console.log(commentContainer);
        if (chatboxContainer.style.display === 'none') {
            let ann = w3c2konva(args.annotation);
            request_url = `${chatAPI}&x0=${ann.x0}&y0=${ann.y0}&x1=${ann.x1}&y1=${ann.y1}&description=${ann.description}`;
            console.log(request_url);

            // If chatbox is empty, generate iframe and append it
            if (chatboxContainer.innerHTML.trim() === '') {
                var iframe = document.createElement('iframe');
                iframe.src = request_url;
                iframe.width = '100%';
                iframe.height = '100%';
                iframe.setAttribute('frameborder', '0');
                chatboxContainer.appendChild(iframe);
            }
            commentContainer.style.display = 'none';
            chatboxContainer.style.display = 'block'; // Show the chatbox
            button.textContent = 'Hide I-Viewer Copilot';
        } else {
            commentContainer.style.display = 'block';
            chatboxContainer.style.display = 'none'; // Hide the chatbox
            button.textContent = 'Open I-Viewer Copilot';
        }
    });

    return container;
}

//     // Add onload event listener to the iframe
//     iframe.onload = function() {
//         // Wait for iframe to load, then send POST request
//         var iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
//         var form = iframeDoc.createElement("form");
//         form.method = "post";
//         form.action = "https://example.com/your-endpoint";
//         form.style.display = "none";

//         // Add form fields
//         postData.split("&").forEach(function(pair) {
//             var keyValue = pair.split("=");
//             var input = iframeDoc.createElement("input");
//             input.type = "hidden";
//             input.name = decodeURIComponent(keyValue[0]);
//             input.value = decodeURIComponent(keyValue[1]);
//             form.appendChild(input);
//         });

//         // Append form to iframe document and submit
//         iframeDoc.body.appendChild(form);
//         form.submit();
//     };