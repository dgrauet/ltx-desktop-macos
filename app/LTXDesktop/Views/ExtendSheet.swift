import SwiftUI

struct ExtendSheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var backendService: BackendService

    let sourceVideoPath: String
    let initialDirection: String

    @State private var prompt = ""
    @State private var direction: String
    @State private var extensionFrames = 49
    @State private var steps = 8
    @State private var seed = -1
    @State private var fps = 24
    @State private var isSubmitting = false
    @State private var errorMessage: String?

    private let frameOptions = [9, 17, 25, 33, 41, 49, 65, 97]

    init(sourceVideoPath: String, initialDirection: String = "forward", fps: Int = 24) {
        self.sourceVideoPath = sourceVideoPath
        self.initialDirection = initialDirection
        _direction = State(initialValue: initialDirection)
        _fps = State(initialValue: fps)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Title bar
            HStack {
                Text("Extend Video")
                    .font(.title2)
                    .fontWeight(.semibold)
                Spacer()
            }
            .padding([.top, .horizontal], 20)
            .padding(.bottom, 16)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Source video
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Source Video")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Text(sourceVideoPath.components(separatedBy: "/").last ?? sourceVideoPath)
                            .font(.caption)
                            .fontWeight(.medium)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color(.controlBackgroundColor).opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 8))

                    // Prompt
                    Text("Prompt")
                        .font(.headline)
                    ZStack(alignment: .topLeading) {
                        if prompt.isEmpty {
                            Text("Describe what happens next (or before). Keep the style and subject consistent with the source video for best results.")
                                .font(.body)
                                .foregroundStyle(.tertiary)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                        }
                        TextEditor(text: $prompt)
                            .frame(minHeight: 80, maxHeight: 120)
                            .font(.body)
                            .scrollContentBackground(.hidden)
                    }
                    .background(Color(.textBackgroundColor).opacity(0.5))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                    )

                    // Direction picker
                    HStack {
                        Text("Direction")
                            .font(.subheadline)
                        Spacer()
                        Picker("", selection: $direction) {
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.right")
                                Text("Forward")
                            }.tag("forward")
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.left")
                                Text("Backward")
                            }.tag("backward")
                        }
                        .frame(width: 140)
                    }

                    // Extension frames
                    HStack {
                        Text("Extension Frames")
                            .font(.subheadline)
                        Spacer()
                        Picker("", selection: $extensionFrames) {
                            ForEach(frameOptions, id: \.self) { n in
                                let secs = String(format: "%.1fs", Double(n) / Double(fps))
                                Text("\(n) (\(secs))").tag(n)
                            }
                        }
                        .frame(width: 130)
                    }

                    // Steps + Seed
                    HStack {
                        Text("Steps")
                            .font(.subheadline)
                        Stepper(value: $steps, in: 1...50) {
                            Text("\(steps)")
                                .monospacedDigit()
                        }

                        Spacer()

                        Text("Seed")
                            .font(.subheadline)
                        TextField("Seed", value: $seed, format: .number)
                            .frame(width: 80)
                            .textFieldStyle(.roundedBorder)
                    }
                    Text("-1 = random seed")
                        .font(.caption2)
                        .foregroundStyle(.secondary)

                    // Error
                    if let error = errorMessage {
                        HStack(spacing: 8) {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                            Text(error)
                                .font(.caption)
                                .foregroundStyle(.red)
                        }
                        .padding(10)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.red.opacity(0.1))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
                .padding(20)
            }

            Divider()

            // Action buttons
            HStack(spacing: 12) {
                Button("Cancel") {
                    dismiss()
                }
                .keyboardShortcut(.escape)

                Spacer()

                Text("Job will appear in the Queue tab")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Button(action: submitExtend) {
                    HStack(spacing: 6) {
                        if isSubmitting {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        Image(systemName: direction == "forward" ? "arrow.right.to.line" : "arrow.left.to.line")
                        Text("Extend")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(prompt.isEmpty || isSubmitting)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(16)
        }
        .frame(width: 480, height: 480)
    }

    private func submitExtend() {
        isSubmitting = true
        errorMessage = nil

        let request = ExtendRequest(
            sourceVideoPath: sourceVideoPath,
            prompt: prompt,
            direction: direction,
            extensionFrames: extensionFrames,
            steps: steps,
            seed: seed,
            fps: fps
        )

        Task {
            do {
                let _ = try await backendService.generateExtend(request: request)
                dismiss()
            } catch {
                errorMessage = "Submit failed: \(error.localizedDescription)"
                isSubmitting = false
            }
        }
    }
}
