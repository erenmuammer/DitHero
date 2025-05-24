document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const fileInput = document.getElementById('fileInput');
    const selectFilesBtn = document.getElementById('selectFilesBtn');
    const fileListUI = document.getElementById('fileList');
    const convertAllBtn = document.getElementById('convertAllBtn');
    const globalStatusBox = document.getElementById('globalStatusBox');
    const noFilesMessage = document.getElementById('noFilesMessage');
    const fileNumInfo = document.getElementById('fileNumInfo');
    const sampleRateSelect = document.getElementById('sampleRateSelect');
    const bitDepthSelect = document.getElementById('bitDepthSelect');
    const dropZone = document.getElementById('drop-zone');
    const ditherNote = document.querySelector('.option-note');
    const mp3BitrateContainer = document.getElementById('mp3BitrateContainer');
    const mp3BitrateSelect = document.getElementById('mp3BitrateSelect');
    const wavOptionsContainer = document.getElementById('wavOptionsContainer');
    const batchDownloadBtn = document.getElementById('batchDownloadBtn');
    const convertToMp3Toggle = document.getElementById('convertToMp3Toggle');

    // --- State ---
    let fileQueue = []; // Array to hold file objects { id, originalFile, status, statusText, element, serverFilepath, originalFilename, analysisInfo, error, downloadUrl, isConverting }
    let isBatchConverting = false;
    let hasMp3Files = false; // Track if we have MP3 files in the queue
    let hasWavFiles = false; // Track if we have WAV files in the queue
    let isZipDownloading = false; // Track if a zip download is in progress
    let convertLosslessToMp3 = false; // Track if we should convert WAV/AIFF/FLAC to MP3

    // --- Event Listeners ---

    // Trigger hidden file input when custom button is clicked
    selectFilesBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle Convert to MP3 toggle
    if (convertToMp3Toggle) {
        convertToMp3Toggle.addEventListener('change', function() {
            convertLosslessToMp3 = this.checked;
            updateOptionsVisibility();
        });
    }

    // Handle file selection
    fileInput.addEventListener('change', async (event) => {
        const files = event.target.files;
        if (!files || files.length === 0) {
            fileNumInfo.textContent = 'No files selected';
            // Optionally reset UI if files are deselected
             resetUI();
            return;
        }

        resetUI(); // Clear previous state and UI elements
        fileNumInfo.textContent = `${files.length} file(s) selected`;
        showGlobalStatus(`Analyzing ${files.length} file(s)...`, 'info');

        const analysisPromises = [];

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const fileId = `file-${Date.now()}-${i}`; // Unique ID

            const fileEntry = {
                id: fileId,
                originalFile: file,
                status: 'analyzing',
                statusText: 'Analyzing...',
                element: null, // Will hold the LI element
                serverFilepath: null,
                originalFilename: file.name,
                analysisInfo: null,
                error: null,
                downloadUrl: null,
                isConverting: false
            };
            fileQueue.push(fileEntry);
            addFileToListUI(fileEntry); // Add placeholder to UI
            analysisPromises.push(analyzeSingleFile(fileEntry)); // Start async analysis
        }

        await Promise.all(analysisPromises);

        // Update global status after all analyses are done (or failed)
        updateUIBasedOnAnalysis();
        
        // Check which file types are in the queue and update option visibility
        updateFileTypeOptions();
    });

    // Handle "Convert All" button click
    convertAllBtn.addEventListener('click', handleBatchConvert);
    
    // Handle "Batch Download" button click
    if (batchDownloadBtn) {
        batchDownloadBtn.addEventListener('click', handleBatchDownload);
    }

    // --- Drag and Drop Event Listeners ---
    dropZone.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover'); // Keep class while hovering
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Check if the leave event is not triggered by entering a child element
        if (e.target === dropZone || !dropZone.contains(e.relatedTarget)) {
            dropZone.classList.remove('dragover');
        }
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length) {
            fileInput.files = files; // Assign dropped files to the hidden input
            // Trigger the change event manually to start processing
            const changeEvent = new Event('change', { bubbles: true });
            fileInput.dispatchEvent(changeEvent);
        }
    });

    // --- Update Dither Note Listener ---
    bitDepthSelect.addEventListener('change', updateDitherNote);
    function updateDitherNote() {
        if (!ditherNote) return;
        const selectedValue = bitDepthSelect.value;
        if (selectedValue === 'float') {
            ditherNote.textContent = 'Dithering is not applied to float format.';
        } else {
            ditherNote.textContent = 'Integer formats include TPDF dither.';
        }
    }
    // Initial call to set the note correctly on load
    updateDitherNote();


    // --- Core Functions ---

    function resetUI() {
        fileQueue = [];
        fileListUI.innerHTML = '';
        noFilesMessage.style.display = 'block'; // Show initially
        globalStatusBox.style.display = 'none';
        globalStatusBox.className = ''; // Clear status classes
        convertAllBtn.disabled = true;
        isBatchConverting = false;
        fileNumInfo.textContent = 'No files selected';
        fileInput.value = ''; // Clear the file input selection
        hasMp3Files = false;
        hasWavFiles = false;
        mp3BitrateContainer.style.display = 'none'; // Hide MP3 bitrate options
        wavOptionsContainer.style.display = 'none'; // Hide WAV options
        
        // Hide batch download button
        if (batchDownloadBtn) {
            batchDownloadBtn.style.display = 'none';
            batchDownloadBtn.disabled = true;
        }
        
        // Reset convert to MP3 toggle (optional)
        if (convertToMp3Toggle && convertToMp3Toggle.checked) {
            convertToMp3Toggle.checked = false;
            convertLosslessToMp3 = false;
        }
    }

    async function analyzeSingleFile(fileEntry) {
        const formData = new FormData();
        formData.append('file', fileEntry.originalFile);

        try {
            const response = await fetch('/analyze', { method: 'POST', body: formData });
            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || `Server error: ${response.status}`);
            }

            // Update fileEntry with analysis results
            fileEntry.status = 'ready';
            fileEntry.statusText = 'Ready';
            fileEntry.serverFilepath = result.filepath;
            fileEntry.analysisInfo = result.info;
            // Use filename from server if available (might be sanitized)
            fileEntry.originalFilename = result.original_filename || fileEntry.originalFilename;
            
            // Check file type based on extension
            const fileExtension = fileEntry.originalFilename.split('.').pop().toLowerCase();
            
            // MP3 files
            if (fileExtension === 'mp3') {
                fileEntry.isMp3 = true;
                fileEntry.isLossy = true;
                hasMp3Files = true;
                
                // Display bitrate info prominently if available
                if (result.info && result.info.bit_rate) {
                    fileEntry.bitrate = result.info.bit_rate;
                    console.log(`Detected bitrate for ${fileEntry.originalFilename}: ${fileEntry.bitrate} kbps`);
                }
            } 
            // WAV and other lossless formats
            else if (['wav', 'aiff', 'aif', 'flac'].includes(fileExtension)) {
                fileEntry.isWav = true;
                fileEntry.isLossless = true;
                hasWavFiles = true;
            }
            // Other lossy formats (ogg, m4a, aac, opus)
            else if (['ogg', 'm4a', 'aac', 'opus'].includes(fileExtension)) {
                fileEntry.isLossy = true;
            }

        } catch (error) {
            console.error('Analysis error for', fileEntry.originalFilename, ':', error);
            fileEntry.status = 'error';
            fileEntry.statusText = 'Analysis Failed';
            fileEntry.error = error.message || 'Unknown analysis error';
        } finally {
             updateFileInListUI(fileEntry); // Update UI for this specific file
        }
    }

    function updateUIBasedOnAnalysis() {
        const readyCount = fileQueue.filter(f => f.status === 'ready').length;
        const errorCount = fileQueue.filter(f => f.status === 'error').length;
        const totalCount = fileQueue.length;

        if (totalCount === 0) {
             resetUI();
             return;
        }

        if (readyCount === 0 && errorCount === totalCount) {
            showGlobalStatus(`Analysis failed for all ${totalCount} files. Please check formats.`, 'danger');
        } else if (errorCount > 0) {
            showGlobalStatus(`Analysis complete. ${readyCount} ready, ${errorCount} error(s) out of ${totalCount}.`, 'warning');
        } else {
            showGlobalStatus(`Analysis complete. ${readyCount} file(s) ready for conversion.`, 'success');
        }
        updateConvertAllButtonState();
    }

    // Function to update options visibility based on file types and settings
    function updateFileTypeOptions() {
        // Show/hide appropriate option containers based on the toggle state and file types
        if (convertLosslessToMp3) {
            // When "Convert to MP3" is enabled, always show MP3 options for both WAV and MP3 files
            mp3BitrateContainer.style.display = 'block';
            // Hide WAV options since we'll be converting to MP3
            wavOptionsContainer.style.display = 'none';
        } else {
            // Standard behavior: show options based on file types
            mp3BitrateContainer.style.display = hasMp3Files ? 'block' : 'none';
            wavOptionsContainer.style.display = hasWavFiles ? 'block' : 'none';
        }
        
        // Add visual separation if needed
        if (hasMp3Files && hasWavFiles && !convertLosslessToMp3) {
            wavOptionsContainer.classList.add('separate-options');
            mp3BitrateContainer.classList.add('separate-options');
        } else {
            wavOptionsContainer.classList.remove('separate-options');
            mp3BitrateContainer.classList.remove('separate-options');
        }
    }

    // Handle toggle visibility changes
    function updateOptionsVisibility() {
        updateFileTypeOptions();
        
        // Update UI feedback based on toggle state
        if (convertLosslessToMp3 && hasWavFiles) {
            showGlobalStatus("WAV/AIFF/FLAC files will be converted to MP3 format.", "info");
        }
    }

    async function handleIndividualConvert(fileEntry) {
        if (fileEntry.status !== 'ready' || fileEntry.isConverting || isBatchConverting) {
            return; // Prevent conversion if not ready, already converting, or batch is running
        }

        fileEntry.isConverting = true;
        fileEntry.status = 'converting';
        fileEntry.statusText = 'Converting...';
        fileEntry.error = null;
        updateFileInListUI(fileEntry);
        updateConvertAllButtonState(); // Disable batch button during individual conversion

        try {
            // Get selected options based on file type
            const conversionData = {
                filepath: fileEntry.serverFilepath,
                original_filename: fileEntry.originalFilename,
            };
            
            // Add proper parameters based on file type and conversion setting
            if (fileEntry.isMp3) {
                // For MP3 files, use the bitrate parameter
                conversionData.mp3_bitrate = parseInt(mp3BitrateSelect.value, 10);
            } else if (fileEntry.isLossless && convertLosslessToMp3) {
                // For lossless files (WAV/AIFF/FLAC) when converting to MP3
                conversionData.target_format = 'mp3';
                conversionData.mp3_bitrate = parseInt(mp3BitrateSelect.value, 10);
            } else {
                // For WAV and other files with regular conversion
                conversionData.target_sr = parseInt(sampleRateSelect.value, 10);
                conversionData.target_bit_depth = bitDepthSelect.value;
            }

            const response = await fetch('/convert', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(conversionData)
            });
            const result = await response.json();
            if (!response.ok) {
                 throw new Error(result.error || `Server error: ${response.status}`);
            }

            fileEntry.status = 'done';
            fileEntry.statusText = 'Completed';
            // Construct download URL based on Flask route (adjust if needed)
            fileEntry.downloadUrl = `/download/${result.download_filename}`;
            
            // Update batch download button state whenever a conversion completes
            updateBatchDownloadButtonState();
            
        } catch (error) {
            console.error('Conversion error for', fileEntry.originalFilename, ':', error);
            fileEntry.status = 'error';
            fileEntry.statusText = 'Conversion Failed';
            fileEntry.error = error.message || 'Unknown conversion error';
        } finally {
            fileEntry.isConverting = false;
            updateFileInListUI(fileEntry);
            updateConvertAllButtonState(); // Re-enable batch button if needed
        }
    }

    async function handleBatchConvert() {
        if (isBatchConverting) return;

        const filesToConvert = fileQueue.filter(f => f.status === 'ready' && !f.isConverting);
        if (filesToConvert.length === 0) {
            showGlobalStatus('No files currently ready for batch conversion.', 'warning');
            return;
        }

        isBatchConverting = true;
        updateConvertAllButtonState(); // Visually disable button and show progress
        convertAllBtn.innerHTML = `<span class="spinner"></span> Converting All (${filesToConvert.length})...`;
        showGlobalStatus(`Starting batch conversion for ${filesToConvert.length} file(s)...`, 'info');

        let batchSuccess = 0;
        let batchError = 0;

        // Get selected options ONCE before the loop
        const targetSr = parseInt(sampleRateSelect.value, 10);
        const targetBitDepth = bitDepthSelect.value; // Keep as string ('16', '24', 'float')
        const targetMp3Bitrate = parseInt(mp3BitrateSelect.value, 10); // Get MP3 bitrate

        // Convert files sequentially to avoid overwhelming the server (can be parallelized if server supports it)
        for (const fileEntry of filesToConvert) {
            // Mark as converting and update UI before awaiting
            fileEntry.isConverting = true; // Still mark individually
            fileEntry.status = 'converting';
            fileEntry.statusText = 'Converting...';
            fileEntry.error = null;
            updateFileInListUI(fileEntry);

            try {
                // Prepare conversion parameters
                const conversionData = {
                    filepath: fileEntry.serverFilepath,
                    original_filename: fileEntry.originalFilename,
                };
                
                // Add proper parameters based on file type and conversion setting
                if (fileEntry.isMp3) {
                    // For MP3 files, use the bitrate parameter
                    conversionData.mp3_bitrate = targetMp3Bitrate;
                } else if (fileEntry.isLossless && convertLosslessToMp3) {
                    // For lossless files (WAV/AIFF/FLAC) when converting to MP3
                    conversionData.target_format = 'mp3';
                    conversionData.mp3_bitrate = targetMp3Bitrate;
                } else {
                    // For WAV and other files with regular conversion
                    conversionData.target_sr = targetSr;
                    conversionData.target_bit_depth = targetBitDepth;
                }

                const response = await fetch('/convert', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(conversionData)
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || `Server error: ${response.status}`);

                fileEntry.status = 'done';
                fileEntry.statusText = 'Completed';
                fileEntry.downloadUrl = `/download/${result.download_filename}`;
                batchSuccess++;
            } catch (error) {
                console.error('Batch conversion error for', fileEntry.originalFilename, ':', error);
                fileEntry.status = 'error';
                fileEntry.statusText = 'Conversion Failed';
                fileEntry.error = error.message || 'Unknown conversion error';
                batchError++;
            } finally {
                fileEntry.isConverting = false; // Mark individual as done processing
                updateFileInListUI(fileEntry); // Update UI for this file
            }
        }

        // Reset batch state and button text after all conversions
        isBatchConverting = false;
        convertAllBtn.innerHTML = 'Convert All Ready'; // Reset button text
        updateConvertAllButtonState(); // Re-evaluate state based on remaining files
        
        // Also explicitly update batch download button
        updateBatchDownloadButtonState();
        
        let finalMessage = `Batch conversion finished. ${batchSuccess} succeeded, ${batchError} failed.`;
        showGlobalStatus(finalMessage, batchError > 0 ? 'warning' : 'success');
    }

    async function handleBatchDownload() {
        if (isZipDownloading) return; // Prevent multiple clicks
        
        // Find all completed files with download URLs
        const completedFiles = fileQueue.filter(f => f.status === 'done' && f.downloadUrl);
        
        if (completedFiles.length === 0) {
            showGlobalStatus('No completed files to download.', 'warning');
            return;
        }
        
        try {
            isZipDownloading = true;
            batchDownloadBtn.innerHTML = '<span class="spinner"></span> Creating ZIP...';
            batchDownloadBtn.disabled = true;
            
            // Extract just the filenames from the download URLs
            const filenames = completedFiles.map(file => {
                // Extract filename from the download URL
                // Format is "/download/filename.ext"
                const parts = file.downloadUrl.split('/');
                return parts[parts.length - 1];
            });
            
            // Send the list of filenames to the server
            const response = await fetch('/download-all', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filenames })
            });
            
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
            
            // Get the ZIP file as a blob
            const blob = await response.blob();
            
            // Create a download link
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `dithero_downloads.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            
            showGlobalStatus(`ZIP file with ${completedFiles.length} files prepared for download.`, 'success');
        } catch (error) {
            console.error('ZIP download error:', error);
            showGlobalStatus(`Error creating ZIP: ${error.message}`, 'danger');
        } finally {
            isZipDownloading = false;
            batchDownloadBtn.innerHTML = 'Download All as ZIP';
            batchDownloadBtn.disabled = false;
            
            // Update button state
            updateBatchDownloadButtonState();
        }
    }

    function updateBatchDownloadButtonState() {
        if (!batchDownloadBtn) return;
        
        // Count completed files
        const completedCount = fileQueue.filter(f => f.status === 'done' && f.downloadUrl).length;
        
        // Show/hide and enable/disable based on completed files
        if (completedCount > 0) {
            batchDownloadBtn.style.display = 'block';
            batchDownloadBtn.disabled = isZipDownloading;
            batchDownloadBtn.innerHTML = isZipDownloading ? 
                '<span class="spinner"></span> Creating ZIP...' : 
                `Download All (${completedCount}) as ZIP`;
        } else {
            batchDownloadBtn.style.display = 'none';
            batchDownloadBtn.disabled = true;
        }
    }

    // --- UI Update Functions ---

    function addFileToListUI(fileEntry) {
        const li = document.createElement('li');
        li.id = fileEntry.id;
        li.innerHTML = createFileListItemHTML(fileEntry); // Initial rendering
        fileEntry.element = li; // Store reference to the element
        fileListUI.appendChild(li);
        attachActionListeners(li, fileEntry); // Attach button listeners
        noFilesMessage.style.display = 'none'; // Hide the 'no files' message
    }

    function updateFileInListUI(fileEntry) {
        if (!fileEntry.element) return;
        // Re-render the list item's content based on the latest fileEntry state
        fileEntry.element.innerHTML = createFileListItemHTML(fileEntry);
        attachActionListeners(fileEntry.element, fileEntry); // Re-attach listeners as innerHTML overwrites them
    }

    function createFileListItemHTML(fileEntry) {
        // Generate the HTML content for a single file list item (<li>)
        // Uses classes defined in style.css

        let infoHTML = '';
        if (fileEntry.error) {
            infoHTML = `<span class="file-error">${escapeHTML(fileEntry.error)}</span>`;
        } else if (fileEntry.analysisInfo) {
            const info = fileEntry.analysisInfo;
            const sr = info.samplerate ? `${info.samplerate} Hz` : 'N/A';
            const depth = info.estimated_bit_depth ? `${info.estimated_bit_depth}-bit` : 'N/A';
            const duration = info.duration_seconds ? `${info.duration_seconds.toFixed(1)}s` : 'N/A';
            const channels = info.channels || 'N/A';
            // Safely access nested format properties
            const format = escapeHTML(info.format || 'N/A');
            const subtype = escapeHTML(info.subtype || 'N/A');

            // Add bitrate information for MP3 files if available
            let bitrate = '';
            if (fileEntry.isMp3) {
                // Convert from bps to kbps if needed
                let bitrateVal = info.bit_rate;
                if (typeof bitrateVal === 'string') {
                    bitrateVal = parseInt(bitrateVal, 10);
                }
                if (bitrateVal > 1000) {
                    bitrateVal = Math.round(bitrateVal / 1000);
                }
                
                // For MP3 files, make bitrate more prominent
                if (bitrateVal) {
                    bitrate = `<span class="highlight" style="color: #007aff; font-weight: bold;">Bitrate: ${bitrateVal} kbps</span>`;
                } else {
                    bitrate = `<span class="highlight">Bitrate: N/A</span>`;
                }
            }

            // For MP3 files, show bitrate information prominently
            if (fileEntry.isMp3) {
                infoHTML = `<span class="file-info">
                               ${bitrate}
                               <span class="highlight">SR: ${sr}</span>
                               <span> | Ch: ${channels}, Dur: ${duration}</span>
                             </span>`;
            } else {
                infoHTML = `<span class="file-info">
                               <span class="highlight">SR: ${sr}</span>
                               <span class="highlight">Depth: ${depth}</span>
                               <span> | Ch: ${channels}, Dur: ${duration}, Format: ${format} ${subtype !== 'N/A' ? `(${subtype})` : ''}</span>
                             </span>`;
            }
        } else if (fileEntry.status === 'analyzing') {
             infoHTML = `<span class="file-info">Awaiting analysis...</span>`;
        }

        let actionButtonHTML = '';
        let removeButtonHTML = '';

        // Show Remove button if file is ready or analysis failed
        if (fileEntry.status === 'ready' || fileEntry.status === 'error') {
             removeButtonHTML = `<button class="action-remove" title="Remove file">&times;</button>`;
        }

        if (fileEntry.status === 'ready') {
            actionButtonHTML = `<button class="action-convert" ${fileEntry.isConverting ? 'disabled' : ''}>
                              ${fileEntry.isConverting ? '<span class="spinner"></span>' : ''} Convert
                            </button>`;
        } else if (fileEntry.status === 'converting') {
             actionButtonHTML = `<button class="action-convert" disabled><span class="spinner"></span> Converting</button>`;
        } else if (fileEntry.status === 'done' && fileEntry.downloadUrl) {
            actionButtonHTML = `<a href="${fileEntry.downloadUrl}" class="action-download" download>Download</a>`;
        }

        // Add spinner to status text if converting
        let statusTextDisplay = escapeHTML(fileEntry.statusText);
        if (fileEntry.status === 'converting' && !fileEntry.isConverting) { // Show spinner in status if batch converting this item
            statusTextDisplay = `<span class="spinner"></span> ${statusTextDisplay}`;
        }


        return `
            <div class="file-details">
                <span class="file-name">${escapeHTML(fileEntry.originalFilename)}</span>
                ${infoHTML}
            </div>
            <div class="file-status status-${fileEntry.status}">${statusTextDisplay}</div>
            <div class="file-actions">
                ${actionButtonHTML}
                ${removeButtonHTML}
            </div>
        `;
    }

    function attachActionListeners(listItemElement, fileEntry) {
        // Attach listener to the 'Convert' button within the list item
        const convertBtn = listItemElement.querySelector('.action-convert');
        if (convertBtn) {
            // Remove previous listener to avoid duplicates if re-attaching
            convertBtn.replaceWith(convertBtn.cloneNode(true)); // Simple way to remove listeners
            listItemElement.querySelector('.action-convert').addEventListener('click', () => handleIndividualConvert(fileEntry));
        }
        // Attach listener for the 'Remove' button
        const removeBtn = listItemElement.querySelector('.action-remove');
        if (removeBtn) {
             removeBtn.replaceWith(removeBtn.cloneNode(true));
             listItemElement.querySelector('.action-remove').addEventListener('click', () => handleRemoveFile(fileEntry));
        }
        // Download links don't need listeners unless for tracking etc.
    }

    function updateConvertAllButtonState() {
        const readyCount = fileQueue.filter(f => f.status === 'ready' && !f.isConverting).length;
        // Disable if batch is running, or any individual file is converting, or no files are ready
        convertAllBtn.disabled = isBatchConverting || fileQueue.some(f => f.isConverting) || readyCount === 0;
        noFilesMessage.style.display = fileQueue.length > 0 ? 'none' : 'block'; // Hide if queue has items
        
        // Also update batch download button state
        updateBatchDownloadButtonState();
    }

    function showGlobalStatus(message, type = 'info') {
        globalStatusBox.textContent = message;
        // Use the CSS classes defined in style.css for status types
        globalStatusBox.className = `status-${type}`; // Reset classes and add the new one
        globalStatusBox.style.display = 'block'; // Make it visible
    }

    // --- Utility Functions ---
    function escapeHTML(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.appendChild(document.createTextNode(str));
        return div.innerHTML;
    }

    // --- Handle File Removal ---
    function handleRemoveFile(fileEntryToRemove) {
        console.log("Removing file:", fileEntryToRemove.originalFilename);

        // Remove from queue array
        fileQueue = fileQueue.filter(entry => entry.id !== fileEntryToRemove.id);

        // Remove from UI
        if (fileEntryToRemove.element) {
            fileEntryToRemove.element.remove();
        }

        // Update UI states (e.g., "Convert All" button, "No Files" message)
        updateConvertAllButtonState();
        if (fileQueue.length === 0) {
             showGlobalStatus('File queue is empty.', 'info');
        }
        // Update file count info
        fileNumInfo.textContent = fileQueue.length > 0 ? `${fileQueue.length} file(s) selected` : 'or drag files here';

        // Note: We don't need to call the backend, as the file hasn't been converted yet.
        // The uploaded file (if analysis succeeded) will be cleaned up by the backend when conversion is attempted or if it fails there.
        // If analysis failed, the backend might have already cleaned it up.
    }

    // --- Initial Setup ---
    resetUI(); // Ensure clean state on load
    
    // Init toggle state
    if (convertToMp3Toggle) {
        convertLosslessToMp3 = convertToMp3Toggle.checked;
    }
    
}); // End DOMContentLoaded 