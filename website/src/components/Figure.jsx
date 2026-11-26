const src = (name) => `${import.meta.env.BASE_URL}images/${name}`;

export default function Figure({ name, alt, caption, wide }) {
  return (
    <figure className={wide ? 'figure wide' : 'figure'}>
      <a href={src(name)} target="_blank" rel="noreferrer">
        <img src={src(name)} alt={alt} loading="lazy" />
      </a>
      {caption && <figcaption>{caption}</figcaption>}
    </figure>
  );
}

export { src as imageSrc };
