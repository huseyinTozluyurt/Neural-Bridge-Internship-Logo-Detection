const REPO = 'https://github.com/huseyinTozluyurt/Neural-Bridge-Internship-Logo-Detection/blob/main/';

export default function FileTag({ path, children }) {
  return (
    <a className="file" href={REPO + path} target="_blank" rel="noreferrer">
      {children || path}
    </a>
  );
}

export { REPO };
