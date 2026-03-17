import SwiftUI

struct RetakeSheet: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject var backendService: BackendService

    let sourceVideoPath: String
    let videoDuration: Double
    let fps: Int

    @State private var prompt = ""
    @State private var startTime: Double = 0
    @State private var endTime: Double = 1
    @State private var steps = 8
    @State private var seed = -1
    @State private var isSubmitting = false
    @State private var errorMessage: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Title bar
            HStack {
                Text("Retake Segment")
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
                    TextEditor(text: $prompt)
                        .frame(minHeight: 80, maxHeight: 120)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .background(Color(.textBackgroundColor).opacity(0.5))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.secondary.opacity(0.3), lineWidth: 1)
                        )

                    // Time range
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Time Range")
                            .font(.headline)

                        // Time range indicator bar
                        GeometryReader { geo in
                            ZStack(alignment: .leading) {
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color(.controlBackgroundColor))
                                    .frame(height: 6)

                                let totalWidth = geo.size.width
                                let startFraction = videoDuration > 0 ? startTime / videoDuration : 0
                                let endFraction = videoDuration > 0 ? endTime / videoDuration : 1
                                RoundedRectangle(cornerRadius: 3)
                                    .fill(Color.accentColor)
                                    .frame(
                                        width: max(0, CGFloat(endFraction - startFraction) * totalWidth),
                                        height: 6
                                    )
                                    .offset(x: CGFloat(startFraction) * totalWidth)
                            }
                        }
                        .frame(height: 6)

                        HStack {
                            Text("Start")
                                .font(.subheadline)
                                .frame(width: 40, alignment: .leading)
                            Slider(value: $startTime, in: 0...max(0.1, videoDuration)) { _ in
                                if startTime >= endTime {
                                    startTime = max(0, endTime - 0.1)
                                }
                            }
                            Text(String(format: "%.1fs", startTime))
                                .font(.subheadline)
                                .monospacedDigit()
                                .frame(width: 50)
                        }

                        HStack {
                            Text("End")
                                .font(.subheadline)
                                .frame(width: 40, alignment: .leading)
                            Slider(value: $endTime, in: 0...max(0.1, videoDuration)) { _ in
                                if endTime <= startTime {
                                    endTime = min(videoDuration, startTime + 0.1)
                                }
                            }
                            Text(String(format: "%.1fs", endTime))
                                .font(.subheadline)
                                .monospacedDigit()
                                .frame(width: 50)
                        }

                        let segmentFrames = Int((endTime - startTime) * Double(fps))
                        Text("\(segmentFrames) frames selected")
                            .font(.caption)
                            .foregroundStyle(.secondary)
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

                Button(action: submitRetake) {
                    HStack(spacing: 6) {
                        if isSubmitting {
                            ProgressView()
                                .scaleEffect(0.7)
                        }
                        Image(systemName: "arrow.triangle.2.circlepath")
                        Text("Retake")
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(prompt.isEmpty || isSubmitting)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding(16)
        }
        .frame(width: 480, height: 520)
        .onAppear {
            endTime = videoDuration
        }
    }

    private func submitRetake() {
        isSubmitting = true
        errorMessage = nil

        let request = RetakeRequest(
            sourceVideoPath: sourceVideoPath,
            prompt: prompt,
            startTimeS: startTime,
            endTimeS: endTime,
            steps: steps,
            seed: seed,
            fps: fps
        )

        Task {
            do {
                let _ = try await backendService.generateRetake(request: request)
                dismiss()
            } catch {
                errorMessage = "Submit failed: \(error.localizedDescription)"
                isSubmitting = false
            }
        }
    }
}
