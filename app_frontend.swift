import SwiftUI
import Photos
import PhotosUI

// MARK: - Models
struct AlbumProcessingResponse: Codable {
    let saved_files: [String]
    let album_vibe_caption: String
    let individual_captions: [String]
    let caption_file: String
}

struct ProcessingRequest: Codable {
    let album_path: String
}

// MARK: - Network Service
class AlbumProcessingService: ObservableObject {
    private let baseURL = "http://localhost:5000"
    
    func processAlbum(albumPath: String) async throws -> AlbumProcessingResponse {
        guard let url = URL(string: "\(baseURL)/api/process_album") else {
            throw NetworkError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody = ProcessingRequest(album_path: albumPath)
        request.httpBody = try JSONEncoder().encode(requestBody)
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw NetworkError.invalidResponse
        }
        
        if httpResponse.statusCode != 200 {
            throw NetworkError.serverError(httpResponse.statusCode)
        }
        
        return try JSONDecoder().decode(AlbumProcessingResponse.self, from: data)
    }
}

enum NetworkError: Error, LocalizedError {
    case invalidURL
    case invalidResponse
    case serverError(Int)
    
    var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .invalidResponse:
            return "Invalid response"
        case .serverError(let code):
            return "Server error: \(code)"
        }
    }
}

// MARK: - Photo Manager
class PhotoManager: ObservableObject {
    @Published var selectedImages: [UIImage] = []
    @Published var photosPickerItems: [PhotosPickerItem] = []
    
    func loadSelectedImages() async {
        var images: [UIImage] = []
        
        for item in photosPickerItems {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                images.append(image)
            }
        }
        
        await MainActor.run {
            self.selectedImages = images
        }
    }
}

// MARK: - Main App View
struct ContentView: View {
    @StateObject private var photoManager = PhotoManager()
    @StateObject private var processingService = AlbumProcessingService()
    @State private var isProcessing = false
    @State private var processingResult: AlbumProcessingResponse?
    @State private var errorMessage: String?
    @State private var showingPhotoPicker = false
    @State private var albumPath = ""
    @State private var showingResults = false
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    headerSection
                    photoSelectionSection
                    albumPathSection
                    if !photoManager.selectedImages.isEmpty {
                        selectedPhotosSection
                    }
                    processButton
                    if let error = errorMessage {
                        errorSection(error)
                    }
                }
                .padding()
            }
            .navigationTitle("Album Processor")
            .navigationBarTitleDisplayMode(.large)
            .sheet(isPresented: $showingResults) {
                ResultsView(result: processingResult)
            }
        }
    }
    
    private var headerSection: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle.angled")
                .font(.system(size: 50))
                .foregroundColor(.blue)
            
            Text("AI Album Processor")
                .font(.title2)
                .fontWeight(.bold)
            
            Text("Select photos or enter album path to generate AI-powered captions and process your best shots")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(.bottom)
    }
    
    private var photoSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Select Photos", systemImage: "photo.stack")
                .font(.headline)
            
            PhotosPicker(
                selection: $photoManager.photosPickerItems,
                maxSelectionCount: 20,
                matching: .images
            ) {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("Choose Photos")
                }
                .foregroundColor(.white)
                .padding()
                .background(Color.blue)
                .cornerRadius(10)
            }
            .onChange(of: photoManager.photosPickerItems) { _ in
                Task {
                    await photoManager.loadSelectedImages()
                }
            }
        }
    }
    
    private var albumPathSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Or Enter Album Path", systemImage: "folder")
                .font(.headline)
            
            TextField("e.g., /path/to/album", text: $albumPath)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .autocapitalization(.none)
                .disableAutocorrection(true)
        }
    }
    
    private var selectedPhotosSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Selected Photos (\(photoManager.selectedImages.count))", systemImage: "checkmark.circle")
                .font(.headline)
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 3), spacing: 8) {
                ForEach(Array(photoManager.selectedImages.enumerated()), id: \.offset) { index, image in
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 100, height: 100)
                        .clipped()
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                        )
                }
            }
        }
    }
    
    private var processButton: some View {
        Button(action: processAlbum) {
            HStack {
                if isProcessing {
                    ProgressView()
                        .scaleEffect(0.8)
                        .foregroundColor(.white)
                }
                Text(isProcessing ? "Processing..." : "Process Album")
            }
            .foregroundColor(.white)
            .padding()
            .frame(maxWidth: .infinity)
            .background(canProcess ? Color.green : Color.gray)
            .cornerRadius(10)
        }
        .disabled(!canProcess || isProcessing)
    }
    
    private var canProcess: Bool {
        !albumPath.isEmpty || !photoManager.selectedImages.isEmpty
    }
    
    private func errorSection(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.red)
            Text(message)
                .foregroundColor(.red)
                .font(.caption)
        }
        .padding()
        .background(Color.red.opacity(0.1))
        .cornerRadius(8)
    }
    
    private func processAlbum() {
        guard canProcess else { return }
        
        Task {
            await MainActor.run {
                isProcessing = true
                errorMessage = nil
            }
            
            do {
                let path = albumPath.isEmpty ? "/tmp/selected_photos" : albumPath
                let result = try await processingService.processAlbum(albumPath: path)
                
                await MainActor.run {
                    processingResult = result
                    showingResults = true
                    isProcessing = false
                }
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isProcessing = false
                }
            }
        }
    }
}

// MARK: - Results View
struct ResultsView: View {
    let result: AlbumProcessingResponse?
    @Environment(\.dismiss) private var dismiss
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if let result = result {
                        albumCaptionSection(result.album_vibe_caption)
                        processedFilesSection(result.saved_files)
                        individualCaptionsSection(result.individual_captions)
                    }
                }
                .padding()
            }
            .navigationTitle("Processing Results")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func albumCaptionSection(_ caption: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Album Vibe Caption", systemImage: "quote.bubble.fill")
                .font(.headline)
                .foregroundColor(.blue)
            
            Text(caption)
                .padding()
                .background(Color.blue.opacity(0.1))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.blue.opacity(0.3), lineWidth: 1)
                )
        }
    }
    
    private func processedFilesSection(_ files: [String]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Processed Files (\(files.count))", systemImage: "doc.fill")
                .font(.headline)
                .foregroundColor(.green)
            
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(files, id: \.self) { file in
                    HStack {
                        Image(systemName: "photo.fill")
                            .foregroundColor(.green)
                        Text(URL(fileURLWithPath: file).lastPathComponent)
                            .font(.caption)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(8)
                }
            }
        }
    }
    
    private func individualCaptionsSection(_ captions: [String]) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Individual Captions", systemImage: "list.bullet")
                .font(.headline)
                .foregroundColor(.orange)
            
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(Array(captions.enumerated()), id: \.offset) { index, caption in
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Photo \(index + 1)")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.orange)
                        Text(caption)
                            .font(.caption)
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .cornerRadius(8)
                }
            }
        }
    }
}

// MARK: - App Entry Point
@main
struct AlbumProcessorApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}