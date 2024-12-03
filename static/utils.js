// Add a button to OSD viewer
function generateButton(viewer) {
    let button = new OpenSeadragon.Button({
        tooltip: 'Display',
        id: 'display_tree',
        srcRest: "../static/images/button_rest.png",
        srcGroup: "../static/images/button_grouphover.png",
        srcHover: "../static/images/button_hover.png",
        srcDown: "../static/images/button_pressed.png",
    });
    viewer.addControl(button.element, { anchor: OpenSeadragon.ControlAnchor.TOP_LEFT });

    return button
}

function parseShape(ann) {
    // poly_x and poly_y are strings of comma-separated floats
    let polyX = ann.poly_x ? ann.poly_x.split(',') :[];
    let polyY = ann.poly_y ? ann.poly_y.split(',') : [];
    if ((!ann.poly_x || ann.poly_x.length === 0) && (!ann.poly_y || ann.poly_y.length === 0)) {// box and point
        let [x0, y0] = [parseFloat(ann.x0), parseFloat(ann.y0)];
        let [x1, y1] = [parseFloat(ann.x1), parseFloat(ann.y1)];
        if (x0 == x1 && y0 == y1) { // point
            return {
                shape: "point",
                x: x0,
                y: y0,
                numPoints: 6,
                innerRadius: 5,
                outerRadius: 10,
            };
        } else { // box
            return {
                shape: "rect",
                x: parseFloat(ann.x0),
                y: parseFloat(ann.y0),
                width: parseFloat(ann.x1) - parseFloat(ann.x0),
                height: parseFloat(ann.y1) - parseFloat(ann.y0),
            };
        }
    } else if (polyX.length === 1 || polyY.length === 1) { // Ellipse and circle
        let itemShape = parseFloat(polyX[0]) === parseFloat(polyY[0]) ? "circle" : "ellipse";
        return {
            shape: itemShape,
            x: parseFloat(ann.xc),
            y: parseFloat(ann.yc),
            radiusX: parseFloat(polyX[0]),
            radiusY: parseFloat(polyY[0]),
        };
    } else { // Polygon
        let points = [];
        for (let i = 0; i < polyX.length && i < polyY.length; i++) {
            points.push(parseFloat(polyX[i]));
            points.push(parseFloat(polyY[i]));
        }
        return {
            shape: "polygon",
            points: points,
            closed: true,
        };
    }
}

function parseDatabaseAnnotation(ann, colorPalette) {
    let item = parseShape(ann);
    let color = colorPalette.getColor(ann.label);

    return {
        id: ann.id,
        label: ann.label || '',
        description: ann.description || '',
        annotator: ann.annotator || '',
        project_id: ann.project_id || '',
        group_id: ann.group_id || '',
        created_at: ann.created_at || '',
        fill: color['face'],
        stroke: color['border'],
        strokeWidth: 2,
        draggable: false,
        ...item,
    }
}

function buildSVGTarget(item) {
    // konva shapes: rect, ellipse, circle, polygon
    console.log("buildSVGTarget", item);
    let value, type;
    let rectflag = false;
    let pointflag = false;
    if (item.shape === "polygon") {
        const pointsString = item.points.reduce((acc, val, index, array) => {
            if (index % 2 === 0) {
                acc += val + ",";
            } else {
                acc += val + " ";
            }
 
            if (index === array.length - 1) {
                acc = acc.trim();
            }
 
            return acc;
        }, "");
        value = `<svg><polygon points="${pointsString}"></polygon></svg>`;
        type = "SvgSelector";
    } else if (item.shape === "ellipse") {
        type = "SvgSelector";
        value = `<svg><ellipse cx="${item.x}" cy="${item.y}" rx="${item.radiusX}" ry="${item.radiusY}"></ellipse></svg>`;
    } else if (item.shape === "circle") {
        type = "SvgSelector";
        value = `<svg><circle cx="${item.x}" cy="${item.y}" r="${item.radiusX}"></circle></svg>`;
    } else if (item.shape === "point") {
        rectflag = true;
        pointflag = true;
        type = "FragmentSelector";
        value = `xywh=pixel:${item.x},${item.y},0,0`;
    } else {
        rectflag = true;
        type = "FragmentSelector";
        value = `xywh=pixel:${item.x},${item.y},${item.width},${item.height}`;
    }

    return {
        "selector": {
            "type": type,
            ...(rectflag ? {"conformsTo": "http://www.w3.org/TR/media-frags/"} : {}),
            "value": value
        },
        ...(pointflag ? {"renderedVia": {"name": "point"}} : {}),
    }
}

function konva2w3c(selectedShape) {
    const { id, label, annotator, description, created_at} = selectedShape.attrs;
    // const bbox = { x0: selectedShape.ann.x0, y0: selectedShape.ann.y0, x1: selectedShape.ann.x1, y1: selectedShape.ann.y1 };

    let labelArray = (label || '').split(',');
    let taggingResult = [];
    if (labelArray.length > 0 && labelArray[0] !== "") {
        for (let item of labelArray) {
            taggingResult.push({
                "type": "TextualBody",
                "value": item.trim(),
                "purpose": "tagging"
            });
        }
    }

    const w3cAnnotation = [
        {
            "type": "Annotation",
            "body": [
                {
                    "type": "TextualBody",
                    "value": description || '',
                    "purpose": "commenting",
                },
                // {
                //     "type": "TextualBody",
                //     "value": '',
                //     "purpose": "replying"
                // },
                ...taggingResult,
                // {
                //     "type" : "",
                //     "value" : bbox,
                //     "purpose" : "aiAssistant"
                // },
                {
                    "type" : "annotator",
                    "value" : annotator,
                    "purpose": "showannotator",
                },
                {
                    "type" : "created_at",
                    "value" : created_at,
                    "purpose": "showtime",
                },
            ],
            "target": buildSVGTarget(selectedShape.attrs),
            "id": `#${id}`
        }
    ];

    return w3cAnnotation;
}

// Extract data from the provided W3C annotation format
function extractSelectorInfo(svgSelector) {
    // Parse the SVG selector to extract polygon points
    if (svgSelector.startsWith('<svg><polygon')) {
        const match = /points="([^"]+)"/.exec(svgSelector);
        const pointsString = match ? match[1] : '';
        // Split the points string and convert to array
        const pointPairs = pointsString.split(' ').flatMap(pair => pair.split(',').map(Number));

        let xArray = [];
        let yArray = [];
        // Extract x and y arrays without rounding
        for (let i = 0; i < pointPairs.length; i++) {
            if (i % 2 === 0) {
                xArray.push(pointPairs[i]);
            } else {
                yArray.push(pointPairs[i]);
            }
        }
        // const xArray = pointPairs.map(point => parseFloat(point[0]));
        // const yArray = pointPairs.map(point => parseFloat(point[1]));
        // Find minX, maxX, minY, maxY as integers
        const minX = Math.floor(Math.min(...xArray));
        const maxX = Math.ceil(Math.max(...xArray));
        const minY = Math.floor(Math.min(...yArray));
        const maxY = Math.ceil(Math.max(...yArray));

        // Create poly_x and poly_y strings with two decimal places
        const polyX = xArray.map(x => (parseFloat(x).toFixed(2)).toString()).join(',');
        const polyY = yArray.map(x => (parseFloat(x).toFixed(2)).toString()).join(',');

        // Create an object with the extracted data
        return {
            poly_x: polyX,
            poly_y: polyY,
            x0: minX,
            y0: minY,
            x1: maxX,
            y1: maxY,
            shape: "polygon"
        };
    } else if (svgSelector.startsWith('<svg><ellipse')) {
        const ellipseMatch = svgSelector.match(/<ellipse cx="(\d+(\.\d+)?)".* cy="(\d+(\.\d+)?)".* rx="(\d+(\.\d+)?)".* ry="(\d+(\.\d+)?)".*<\/ellipse>/);
        if (ellipseMatch) {
            const [, cx, , cy, , rx, , ry] = ellipseMatch;
            const parsedXC = parseFloat(cx);
            const parsedYC = parseFloat(cy);
            const parsedRX = parseFloat(rx);
            const parsedRY = parseFloat(ry);
            return {
                poly_x: parsedRX.toFixed(2),
                poly_y: parsedRY.toFixed(2),
                x0: (parsedXC - parsedRX).toFixed(2),
                y0: (parsedYC - parsedRY).toFixed(2),
                x1: (parsedXC + parsedRX).toFixed(2),
                y1: (parsedYC + parsedRY).toFixed(2),
                xc: parsedXC.toFixed(2),
                yc: parsedYC.toFixed(2),
                shape: "ellipse"
            };
        }
    } else if (svgSelector.startsWith('<svg><circle')) {
        // Circle format: <circle cx="x" cy="y" r="radius"></circle>
        const circleMatch = svgSelector.match(/<circle cx="(\d+(\.\d+)?)".* cy="(\d+(\.\d+)?)".* r="(\d+(\.\d+)?)".*<\/circle>/);
        if (circleMatch) {
            const [, cx, , cy, , r] = circleMatch;
            const parsedXC = parseFloat(cx);
            const parsedYC = parseFloat(cy);
            const parsedR = parseFloat(r);
            return {
                poly_x: parseFloat(r).toFixed(2),
                poly_y: parseFloat(r).toFixed(2),
                x0: (parsedXC - parsedR).toFixed(2),
                y0: (parsedYC - parsedR).toFixed(2),
                x1: (parsedXC + parsedR).toFixed(2),
                y1: (parsedYC + parsedR).toFixed(2),
                xc: parseFloat(cx).toFixed(2),
                yc: parseFloat(cy).toFixed(2),
                shape: "circle"
            };
        }
    } else if (svgSelector.startsWith('xywh=pixel:')) {
        const rectMatch = svgSelector.match(/xywh=pixel:(\d+(\.\d+)?),(\d+(\.\d+)?),(\d+(\.\d+)?),(\d+(\.\d+)?)/);
        if (rectMatch) {
            const [, x, , y, , width, , height] = rectMatch;
            const parsedX = parseFloat(x);
            const parsedY = parseFloat(y);
            const parsedWidth = parseFloat(width);
            const parsedHeight = parseFloat(height);
            if (parsedWidth == 0 && parsedHeight == 0) {
                return {
                    poly_x: "",
                    poly_y: "",
                    x0: parsedX.toFixed(2),
                    y0: parsedY.toFixed(2),
                    x1: (parsedX + parsedWidth).toFixed(2),
                    y1: (parsedY + parsedHeight).toFixed(2),
                    shape: "point"
                };
            } else {
                return {
                    poly_x: "",
                    poly_y: "",
                    x0: parsedX.toFixed(2),
                    y0: parsedY.toFixed(2),
                    x1: (parsedX + parsedWidth).toFixed(2),
                    y1: (parsedY + parsedHeight).toFixed(2),
                    shape: "rect"
                };
            }
        }
    } else if (svgSelector.startsWith('<svg><path')) {
        const pathMatch = svgSelector.match(/<path d="([^"]*)".*<\/path>/);
        if (pathMatch) {
            const pathData = pathMatch[1];
            // Parse the path data into an array of commands
            const pathCommands = pathData.split(/(?=[MLCZ])/).filter(Boolean);

            // Extract x and y coordinates from path commands
            const polyX = [];
            const polyY = [];

            pathCommands.forEach(command => {
                const params = command.substring(1).split(' ').map(parseFloat);
                polyX.push(parseFloat(params[0]).toFixed(2));
                polyY.push(parseFloat(params[1]).toFixed(2));
            });
            const minX = Math.floor(Math.min(...polyX));
            const maxX = Math.ceil(Math.max(...polyX));
            const minY = Math.floor(Math.min(...polyY));
            const maxY = Math.ceil(Math.max(...polyY));

            return {
                poly_x: polyX.join(','),
                poly_y: polyY.join(','),
                x0: minX,
                y0: minY,
                x1: maxX,
                y1: maxY,
                shape: "freehand"
            };
        }
    } else {
        return {};
    }
}

function extractAnnotationInfo(w3cAnnotation) {
    let description = "";
    var labels = [];
    w3cAnnotation.body.find(function(b) {
        if (b.purpose == 'tagging') {
            labels.push(b.value);
        }
    })

    var currentDes = w3cAnnotation ? w3cAnnotation.body.filter(function(b) {
        return b.purpose == 'commenting';
    }) : [];
    var currentReplies = w3cAnnotation ? w3cAnnotation.body.filter(function(b) {
        return b.purpose == 'replying';
    }) : [];

    // console.log("length", currentDes.length, currentReplies.length);
    if (currentDes && currentDes.length > 0) {
        description += currentDes.map(function(item) {
            // console.log(item.value)
            return item.value;
        }).join('\n');
    }
    if (currentDes.length > 0 && currentReplies.length > 0) {
        description += '\n';
    }
    if (currentReplies.length > 0) {
        description += currentReplies.map(function(reply) {
            return reply.value;
        }).join('\n');
    }

    // if(w3cAnnotation.body.length>0){
    //     description = w3cAnnotation.body[0].value;
    // }

    return {'label': labels.join(','), 'description': description}
}

function w3c2konva(annotation) {
    let svgSelector = annotation ? annotation.target.selector.value : null;
    let query = extractSelectorInfo(svgSelector);  // JSON.stringify(annotation)
    let annInfo = extractAnnotationInfo(annotation);
    query['label'] = annInfo.label;
    query['description'] = annInfo.description;

    return query;
}

function refineSelection(api, query) {
    console.log("refineSelection", api, query);
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function createAnnotation(api, query) {
    console.log("createAnnotation", api, query);
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function readAnnotation(api) {
    console.log("readAnnotation", api);
    return fetch(api, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function updateAnnotation(api, query) {
    console.log("updateAnnotation", api, query);
    return fetch(api, {
        method: 'PUT',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function deleteAnnotation(api) {
    console.log("deleteAnnotation", api);
    return fetch(api, {
        method: 'DELETE',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    }).catch(error => {
        console.error('There was a problem with the delete operation:', error);
        throw error;
    });
}

function searchAnnotations(api, query) {
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        return response.ok ? response.json() : null;
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function countAnnotations(api) {
    return fetch(api, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(query)
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function getAnnotators(api) {
    return fetch(api, {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        },
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function getAIDescription(api, query) {
    console.log("aiGeneration", api, query);
    let queryString = Object.keys(query).map(key => `${encodeURIComponent(key)}=${encodeURIComponent(query[key])}`).join('&');
    let url = `${api}&${queryString}`;

    return fetch(url, {
        method: 'GET',
    }).then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.text();
    }).catch(error => {
        console.error('There was a problem with the fetch operation:', error);
        throw error;
    });
}

function getCurrentViewport(viewer) {
    // Get the current viewport rectangle
    if (!viewer.world.getItemCount()) {
        return null;
    }
    let slideImage = viewer.world.getItemAt(0);
    let viewportRect = viewer.viewport.getBounds(true);
    let imageTopLeft = slideImage.viewportToImageCoordinates(
        viewportRect.x, viewportRect.y
    );
    let imageBottomRight = slideImage.viewportToImageCoordinates(
        viewportRect.x + viewportRect.width, viewportRect.y + viewportRect.height
    );
    let bbox = {
        'x0': imageTopLeft.x,
        'y0': imageTopLeft.y,
        'x1': imageBottomRight.x,
        'y1': imageBottomRight.y,
    };
    // console.log('Bbox:', bbox, this._viewer.viewport.getZoom(), viewer.viewport.getCenter());
    return bbox;
}

function drawNUpdateDatatable(api, query){
    if (query.annotator.length > 0) { //only draw rows with selected manual annotator in the table
        searchAnnotations(api, query).then(ann => {
            if ($.fn.DataTable.isDataTable('#dataTable')) {
                $('#dataTable').DataTable().clear().destroy();
            }
            if (ann) { // if return is not null
                ann.reverse();
                var data = ann.map(annotation => [
                    userIdNameMap[annotation.annotator], 
                    annotation.description,
                    annotation.label,
                    `[${annotation.x0}, ${annotation.x1}, ${annotation.y0}, ${annotation.y1}]`,
                    annotation
                ]);
                $('#dataTable').DataTable({
                    data: data,
                    scrollX: true,
                    columns: [
                        { title: 'Annotator' },
                        { title: 'Description' },
                        { title: 'Label' },
                        { title: 'Coordinates' },
                        { title: 'Zoom to Annotation', orderable: false } // New column for actions
                    ],
                    columnDefs: [{
                        targets: -1, 
                        render: function (data, type, row, meta) {
                            return '<i class="fas fa-search zoom-to-annotation"></i>'; 
                        }
                    }],
                    dom: 'Bfrtip',
                    buttons: [
                        {
                            text: '<i class="fa fa-download" style="margin-right: 5px"></i>Download as CSV',
                            className: 'btn-sm btn-dt',
                            action: function (e, dt, button, config) {
                                downloadAllAsCSV(ann); // Call function to download all annotations
                            }
                        }, {
                            text: '<i class="fas fa-sync-alt" style="margin-right: 5px"></i>Reload Table',
                            className: 'btn-sm btn-dt',
                            action: function (e, dt, button, config) {
                                drawNUpdateDatatable(api, query); // Call function to download all annotations
                            }
                        }
                    ]
                });

                $('#dataTable tbody').on('click', 'td:nth-child(5)', function () {
                    var coordinateStr = $(this).closest('tr').find('td:nth-child(4)').text(); // Get the text content of the clicked cell
                    var coordinates = coordinateStr.substring(1, coordinateStr.length - 1).split(', '); // Parse coordinates from string
                    var x0 = parseFloat(coordinates[0]);
                    var x1 = parseFloat(coordinates[1]);
                    var y0 = parseFloat(coordinates[2]);
                    var y1 = parseFloat(coordinates[3]);

                    var viewportTopLeft = leftViewer.viewport.imageToViewportCoordinates(
                        new OpenSeadragon.Point(x0, y0),
                    );

                    var viewportBottomRight = leftViewer.viewport.imageToViewportCoordinates(
                        new OpenSeadragon.Point(x1, y1)
                    );

                    var bounds = new OpenSeadragon.Rect(
                        viewportTopLeft.x,
                        viewportTopLeft.y,
                        viewportBottomRight.x - viewportTopLeft.x,
                        viewportBottomRight.y - viewportTopLeft.y
                    );
                    leftViewer.viewport.fitBounds(bounds, true); 
                    leftViewer.viewport.zoomBy(0.9);

                    // var selected = annotationLayer_left._annotoriousLayer.getSelected();
                    // if(!selected){
                    //     annotationLayer_left._annotoriousLayer.removeAnnotation(annotation);
                    //     annotationLayer_left._annotoriousLayer.cancelSelected();
                    //     annotationLayer_left._annotoriousLayer.clearAnnotations();
                    //     var rowIndex = $(this).closest('tr').index();
                    //     var table = $('#dataTable').DataTable();
                    //     var rowData = table.row(rowIndex).data();
                    //     console.log("rowdata",rowData)
                    //     var konvaItem = parseDatabaseAnnotation(rowData[4], colorPalette);
                    //     var svgAnno = object2w3c(konvaItem)[0];
                    //     annotationLayer_left._annotoriousLayer.addAnnotation(svgAnno);
                    //     annotationLayer_left._annotoriousLayer.selectAnnotation(svgAnno);
                    // }
                });
            }
        });
    } else {
        if ($.fn.DataTable.isDataTable('#dataTable')) {
            $('#dataTable').DataTable().clear().destroy();
        }
    }
}

function downloadAllAsCSV(annotations) {
    // Convert annotations to CSV format
    var headerRow = ['Annotator', 'Description', 'Label', 
                     'x0', 'y0', 'x1', 'y1', 'xc', 'yc',
                     'poly_x', 'poly_y'];

    var csvData = annotations.map(function (annotation) {
        return [
            userIdNameMap[annotation.annotator], 
            annotation.description, 
            annotation.label, 
            annotation.x0, 
            annotation.y0, 
            annotation.x1, 
            annotation.y1,
            annotation.xc, 
            annotation.yc,
            annotation.poly_x,
            annotation.poly_y
        ].map(escapeCSV).join(',');
    });

    // var headerString = '"' + headerRow.join('","') + '"';
    // var csvDataString = csvData.join('\n');
    // var csvString = headerString + '\n' + csvDataString;
    var csvString = ['"' + headerRow.join('","') + '"'].concat(csvData).join('\n');
    // Create a Blob containing the CSV data
    var blob = new Blob([csvString], { type: 'text/csv' });

    // Create a temporary URL for the Blob
    var url = window.URL.createObjectURL(blob);

    // Create a link element and trigger the download
    var a = document.createElement('a');
    a.href = url;
    a.download = 'annotations.csv';
    document.body.appendChild(a);
    a.click();

    // Clean up
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

function escapeCSV(text) {
    // If the text contains double quotes, escape them by doubling them
    text = String(text);
    if (text.includes('"')) {
        text = text.replace(/"/g, '""');
    }
    text = `"${text}"`;
    return text;
}

function sendMessageWithRetry(webSocket, query) {
    const MAX_RETRIES = 3; 
    const RETRY_INTERVAL = 3000; 
    let retries = 0;
    function retrySend() {
        if (retries < MAX_RETRIES) {
            try {
                webSocket.send(JSON.stringify(query));

            } catch (error) {
                console.log('Error sending message:', error);
                retries++;
                setTimeout(retrySend, RETRY_INTERVAL);
            }
        } else {
            console.error('Maximum retry attempts reached. Message sending failed.');
        }
    }
    retrySend();
}

function buildConnections(layer, annoAPI) {
    layer.buildConnections(
        annoAPI.createDB, annoAPI.getAnnotator, annoAPI.getLabels,
        annoAPI.insert, annoAPI.read, annoAPI.update, annoAPI.delete,
        annoAPI.search, annoAPI.stream, annoAPI.countAnnos
    );
}

// Function to merge global palette with user-specific palette
function mergeColorPalettes(globalPalette, userPalette) {
    return Object.keys(userPalette).length > 0 ? userPalette : globalPalette;
}

// Function to fetch user-specific color palette
async function fetchColorPalette(imageId, userId) {
    return ""
}